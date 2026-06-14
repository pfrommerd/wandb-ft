use std::collections::HashMap;
use std::time::SystemTime;

use graphql_client::GraphQLQuery;
use serde::Serialize;

use crate::error::{ApiError, ReqwestBadResponse};
use crate::gql::{create_run_files, CreateRunFiles};
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

/// Get the current time in UNIX seconds.
fn current_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("System time was before the UNIX epoch")
        .as_secs_f64()
}
