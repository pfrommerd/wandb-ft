use std::collections::HashMap;
use std::time::SystemTime;

use base64::{Engine, prelude::BASE64_STANDARD as base64};
use graphql_client::GraphQLQuery;
use serde::Serialize;
use serde_json::json;

use crate::artifact::{ArtifactEntrySource, ArtifactSnapshot};
use crate::error::{ApiError, ReqwestBadResponse};
use crate::gql::{
    CommitArtifact, CreateArtifact, CreateArtifactFiles, CreateArtifactManifest, CreateRunFiles,
    create_artifact, create_artifact_files, create_artifact_manifest, create_run_files,
};
use crate::media::ParsedRow;

const TIMESTAMP_METRIC_NAME: &str = "_timestamp";

#[derive(Debug, Serialize)]
struct FsChunkData {
    content: Vec<String>,
    offset: u64,
}

/// A filestream request body. Ported from `wandb-rs`, extended with the optional
/// `complete`/`exitcode` fields used to mark a run finished.
#[derive(Debug, Serialize)]
struct FsFilesData {
    files: HashMap<String, FsChunkData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    complete: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exitcode: Option<i32>,
}

/// Shared, thread-safe state backing a `Run`.
///
/// `client` carries the wandb auth headers (used for GraphQL + filestream).
/// `upload_client` is header-free, used for presigned media file PUTs (an extra
/// `Authorization` header would break those signed requests).
pub struct RunState {
    client: reqwest::Client,
    upload_client: reqwest::Client,
    base_url: String,
    entity: String,
    project: String,
    name: String,
}

impl RunState {
    pub fn new(
        client: reqwest::Client,
        upload_client: reqwest::Client,
        base_url: String,
        entity: String,
        project: String,
        name: String,
    ) -> Self {
        Self {
            client,
            upload_client,
            base_url,
            entity,
            project,
            name,
        }
    }

    fn run_path(&self) -> String {
        format!(
            "{}/files/{}/{}/{}/file_stream",
            self.base_url, self.entity, self.project, self.name
        )
    }

    /// Log one row of metrics (and any media) at the given step.
    pub async fn log(&self, row: ParsedRow, step: u64) -> Result<(), ApiError> {
        let ParsedRow { mut scalars, media } = row;

        if !media.is_empty() {
            // Build the upload list and the `_type` records, upload, then fold
            // the records into the row so they land in history/summary.
            let mut uploads: Vec<(String, Vec<u8>)> = Vec::with_capacity(media.len());
            let mut records: Vec<(String, crate::data_value::DataValue)> =
                Vec::with_capacity(media.len());
            for (key, file) in &media {
                let (path, record) = file.record(key, step);
                uploads.push((path, file.bytes().to_vec()));
                records.push((key.clone(), record));
            }
            self.upload_files(&uploads).await?;
            for (key, record) in records {
                scalars.insert(key, record);
            }
        }

        scalars.insert_default(TIMESTAMP_METRIC_NAME, current_timestamp());

        let row_string = serde_json::to_string(&scalars)?;
        let body = FsFilesData {
            files: [
                (
                    "wandb-history.jsonl".to_string(),
                    FsChunkData {
                        content: vec![row_string.clone()],
                        offset: step,
                    },
                ),
                (
                    "wandb-summary.json".to_string(),
                    FsChunkData {
                        content: vec![row_string],
                        offset: 0,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            complete: None,
            exitcode: None,
        };

        self.client
            .post(self.run_path())
            .json(&body)
            .send()
            .await?
            .maybe_err()
            .await?;
        Ok(())
    }

    /// Upload media files: request presigned URLs via `CreateRunFiles`, then PUT
    /// each file's bytes (with the returned upload headers) to its URL.
    async fn upload_files(&self, files: &[(String, Vec<u8>)]) -> Result<(), ApiError> {
        let variables = create_run_files::Variables {
            entity: self.entity.clone(),
            project: self.project.clone(),
            run: self.name.clone(),
            files: files.iter().map(|(p, _)| p.clone()).collect(),
        };
        let request_body = CreateRunFiles::build_query(variables);

        let mut res: graphql_client::Response<create_run_files::ResponseData> = self
            .client
            .post(format!("{}/graphql", self.base_url))
            .json(&request_body)
            .send()
            .await?
            .maybe_err()
            .await?
            .json()
            .await?;
        if let Some(errors) = &mut res.errors {
            if !errors.is_empty() {
                return Err(ApiError::QueryFailed(errors.drain(..).collect()));
            }
        }
        let payload = res
            .data
            .and_then(|d| d.create_run_files)
            .ok_or_else(|| ApiError::NoResponse("CreateRunFiles returned no data".into()))?;

        let urls: HashMap<String, Option<String>> = payload
            .files
            .into_iter()
            .map(|f| (f.name, f.upload_url))
            .collect();

        for (path, bytes) in files {
            let url = urls
                .get(path)
                .and_then(|u| u.clone())
                .ok_or_else(|| ApiError::NoResponse(format!("no upload URL for {path}")))?;
            let mut req = self.upload_client.put(&url).body(bytes.clone());
            for header in &payload.upload_headers {
                if let Some((k, v)) = header.split_once(':') {
                    req = req.header(k.trim(), v.trim());
                }
            }
            req.send().await?.maybe_err().await?;
        }
        Ok(())
    }

    /// Replace the run summary file with a JSON object.
    pub async fn update_summary(&self, summary: String) -> Result<(), ApiError> {
        let body = FsFilesData {
            files: [(
                "wandb-summary.json".to_string(),
                FsChunkData {
                    content: vec![summary],
                    offset: 0,
                },
            )]
            .into_iter()
            .collect(),
            complete: None,
            exitcode: None,
        };

        self.client
            .post(self.run_path())
            .json(&body)
            .send()
            .await?
            .maybe_err()
            .await?;
        Ok(())
    }

    /// Create, upload, and commit an artifact.
    pub async fn log_artifact(
        &self,
        artifact: ArtifactSnapshot,
        aliases: Option<Vec<String>>,
    ) -> Result<(), ApiError> {
        let files = materialize_artifact_files(&artifact).await?;
        let initial_digest = digest_bytes(artifact.name.as_bytes());
        let artifact_id = self
            .create_artifact(&artifact, aliases, initial_digest)
            .await?;
        let uploaded = self
            .create_and_upload_artifact_files(&artifact_id, &files)
            .await?;
        let manifest = build_manifest(&uploaded);
        let manifest_bytes = serde_json::to_vec(&manifest)?;
        let manifest_digest = digest_bytes(&manifest_bytes);
        self.create_and_upload_artifact_manifest(&artifact_id, &manifest_digest, manifest_bytes)
            .await?;
        self.commit_artifact(&artifact_id).await?;
        Ok(())
    }

    async fn create_artifact(
        &self,
        artifact: &ArtifactSnapshot,
        aliases: Option<Vec<String>>,
        digest: String,
    ) -> Result<String, ApiError> {
        let aliases = aliases.unwrap_or_else(|| vec!["latest".to_string()]);
        let variables = create_artifact::Variables {
            input: create_artifact::CreateArtifactInput {
                aliases: Some(
                    aliases
                        .into_iter()
                        .map(|alias| create_artifact::ArtifactAliasInput {
                            alias,
                            artifact_collection_name: artifact.name.clone(),
                        })
                        .collect(),
                ),
                artifact_collection_name: Some(artifact.name.clone()),
                artifact_collection_names: None,
                artifact_type_name: artifact.type_name.clone(),
                client_id: Some(unique_client_id(&artifact.name)),
                client_mutation_id: None,
                description: artifact.description.clone(),
                digest,
                digest_algorithm: create_artifact::ArtifactDigestAlgorithm::MANIFEST_MD5,
                distributed_id: None,
                enable_digest_deduplication: Some(true),
                entity_name: self.entity.clone(),
                history_step: None,
                labels: None,
                metadata: None,
                project_name: self.project.clone(),
                run_name: Some(self.name.clone()),
                sequence_client_id: Some(unique_client_id(&format!("{}-sequence", artifact.name))),
                tags: None,
                ttl_duration_seconds: None,
            },
        };
        let request_body = CreateArtifact::build_query(variables);
        let mut res: graphql_client::Response<create_artifact::ResponseData> = self
            .client
            .post(format!("{}/graphql", self.base_url))
            .json(&request_body)
            .send()
            .await?
            .maybe_err()
            .await?
            .json()
            .await?;
        if let Some(errors) = &mut res.errors {
            if !errors.is_empty() {
                return Err(ApiError::QueryFailed(errors.drain(..).collect()));
            }
        }
        res.data
            .and_then(|d| d.create_artifact)
            .map(|p| p.artifact.id)
            .ok_or_else(|| ApiError::NoResponse("CreateArtifact returned no artifact id".into()))
    }

    async fn create_and_upload_artifact_files(
        &self,
        artifact_id: &str,
        files: &[ArtifactUploadFile],
    ) -> Result<Vec<UploadedArtifactFile>, ApiError> {
        let variables = create_artifact_files::Variables {
            artifact_files: files
                .iter()
                .map(|file| create_artifact_files::CreateArtifactFileSpecInput {
                    artifact_id: artifact_id.to_string(),
                    artifact_manifest_id: None,
                    md5: file.digest.clone(),
                    mimetype: None,
                    name: file.name.clone(),
                    upload_parts_input: None,
                })
                .collect(),
            storage_layout: create_artifact_files::ArtifactStorageLayout::V2,
        };
        let request_body = CreateArtifactFiles::build_query(variables);
        let mut res: graphql_client::Response<create_artifact_files::ResponseData> = self
            .client
            .post(format!("{}/graphql", self.base_url))
            .json(&request_body)
            .send()
            .await?
            .maybe_err()
            .await?
            .json()
            .await?;
        if let Some(errors) = &mut res.errors {
            if !errors.is_empty() {
                return Err(ApiError::QueryFailed(errors.drain(..).collect()));
            }
        }
        let edges = res
            .data
            .and_then(|d| d.create_artifact_files)
            .map(|p| p.files.edges)
            .ok_or_else(|| ApiError::NoResponse("CreateArtifactFiles returned no files".into()))?;

        let mut uploaded = Vec::with_capacity(files.len());
        for (file, edge) in files.iter().zip(edges.into_iter()) {
            let node = edge.node.ok_or_else(|| {
                ApiError::NoResponse("CreateArtifactFiles returned an empty file node".into())
            })?;
            let upload_url = node.upload_url.ok_or_else(|| {
                ApiError::NoResponse(format!(
                    "CreateArtifactFiles returned no upload URL for {}",
                    file.name
                ))
            })?;
            let headers = node.upload_headers;
            self.upload_bytes(&upload_url, &headers, file.bytes.clone())
                .await?;
            uploaded.push(UploadedArtifactFile {
                name: file.name.clone(),
                digest: file.digest.clone(),
                size: file.bytes.len() as u64,
                storage_path: node.storage_path.ok_or_else(|| {
                    ApiError::NoResponse(format!(
                        "CreateArtifactFiles returned no storage path for {}",
                        file.name
                    ))
                })?,
            });
        }
        Ok(uploaded)
    }

    async fn create_and_upload_artifact_manifest(
        &self,
        artifact_id: &str,
        digest: &str,
        bytes: Vec<u8>,
    ) -> Result<(), ApiError> {
        let variables = create_artifact_manifest::Variables {
            artifact_id: artifact_id.to_string(),
            base_artifact_id: None,
            name: "wandb_manifest.json".to_string(),
            digest: digest.to_string(),
            entity_name: self.entity.clone(),
            project_name: self.project.clone(),
            run_name: self.name.clone(),
            manifest_type: create_artifact_manifest::ArtifactManifestType::FULL,
            include_upload: true,
        };
        let request_body = CreateArtifactManifest::build_query(variables);
        let mut res: graphql_client::Response<create_artifact_manifest::ResponseData> = self
            .client
            .post(format!("{}/graphql", self.base_url))
            .json(&request_body)
            .send()
            .await?
            .maybe_err()
            .await?
            .json()
            .await?;
        if let Some(errors) = &mut res.errors {
            if !errors.is_empty() {
                return Err(ApiError::QueryFailed(errors.drain(..).collect()));
            }
        }
        let file = res
            .data
            .and_then(|d| d.create_artifact_manifest)
            .map(|p| p.artifact_manifest.file)
            .ok_or_else(|| {
                ApiError::NoResponse("CreateArtifactManifest returned no file".into())
            })?;
        let upload_url = file.upload_url.ok_or_else(|| {
            ApiError::NoResponse("CreateArtifactManifest returned no upload URL".into())
        })?;
        let headers = file.upload_headers;
        self.upload_bytes(&upload_url, &headers, bytes).await
    }

    async fn commit_artifact(&self, artifact_id: &str) -> Result<(), ApiError> {
        let variables = crate::gql::commit_artifact::Variables {
            artifact_id: artifact_id.to_string(),
        };
        let request_body = CommitArtifact::build_query(variables);
        let mut res: graphql_client::Response<crate::gql::commit_artifact::ResponseData> = self
            .client
            .post(format!("{}/graphql", self.base_url))
            .json(&request_body)
            .send()
            .await?
            .maybe_err()
            .await?
            .json()
            .await?;
        if let Some(errors) = &mut res.errors {
            if !errors.is_empty() {
                return Err(ApiError::QueryFailed(errors.drain(..).collect()));
            }
        }
        Ok(())
    }

    async fn upload_bytes(
        &self,
        upload_url: &str,
        headers: &[String],
        bytes: Vec<u8>,
    ) -> Result<(), ApiError> {
        let mut req = self.upload_client.put(upload_url).body(bytes);
        for header in headers {
            if let Some((k, v)) = header.split_once(':') {
                req = req.header(k.trim(), v.trim());
            }
        }
        req.send().await?.maybe_err().await?;
        Ok(())
    }

    /// Mark the run finished on the backend.
    pub async fn finish(&self) -> Result<(), ApiError> {
        let body = FsFilesData {
            files: HashMap::new(),
            complete: Some(true),
            exitcode: Some(0),
        };

        self.client
            .post(self.run_path())
            .json(&body)
            .send()
            .await?
            .maybe_err()
            .await?;
        Ok(())
    }
}

struct ArtifactUploadFile {
    name: String,
    digest: String,
    bytes: Vec<u8>,
}

struct UploadedArtifactFile {
    name: String,
    digest: String,
    size: u64,
    storage_path: String,
}

async fn materialize_artifact_files(
    artifact: &ArtifactSnapshot,
) -> Result<Vec<ArtifactUploadFile>, ApiError> {
    let mut files = Vec::with_capacity(artifact.entries.len());
    for entry in &artifact.entries {
        let bytes = match &entry.source {
            ArtifactEntrySource::File(path) => std::fs::read(path)?,
            ArtifactEntrySource::Bytes(bytes) => bytes.clone(),
        };
        files.push(ArtifactUploadFile {
            name: entry.name.clone(),
            digest: digest_bytes(&bytes),
            bytes,
        });
    }
    Ok(files)
}

fn build_manifest(files: &[UploadedArtifactFile]) -> serde_json::Value {
    let contents = files
        .iter()
        .map(|file| {
            (
                file.name.clone(),
                json!({
                    "digest": file.digest,
                    "size": file.size,
                    "path": file.storage_path,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    json!({
        "version": 1,
        "storagePolicy": "wandb-storage-policy-v1",
        "storagePolicyConfig": {
            "storageLayout": "V2",
        },
        "contents": contents,
    })
}

fn digest_bytes(bytes: &[u8]) -> String {
    base64.encode(md5::compute(bytes).0)
}

fn unique_client_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("System time was before the UNIX epoch")
        .as_nanos();
    format!("wandb-ft-{prefix}-{nanos}")
}

/// Get the current time in UNIX seconds.
fn current_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("System time was before the UNIX epoch")
        .as_secs_f64()
}
