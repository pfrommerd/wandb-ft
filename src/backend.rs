use base64::{prelude::BASE64_STANDARD as base64, Engine};
use graphql_client::GraphQLQuery;

use crate::error::{ApiError, ReqwestBadResponse};
use crate::gql::{upsert_bucket, UpsertBucket};

const DEFAULT_API_URL: &str = "https://api.wandb.ai";

/// Owns the authenticated HTTP client and base URL for talking to the wandb
/// backend. Ported from `wandb-rs`'s `WandB`.
pub struct Backend {
    client: reqwest::Client,
    upload_client: reqwest::Client,
    base_url: String,
}

impl Backend {
    pub fn new(api_key: &str) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Basic {}", base64.encode(format!("api:{api_key}")))
                .parse()
                .unwrap(),
        );
        headers.insert(reqwest::header::USER_AGENT, "wandb-core".parse().unwrap());
        Self {
            client: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .unwrap(),
            // A header-free client for presigned media uploads.
            upload_client: reqwest::Client::new(),
            base_url: DEFAULT_API_URL.into(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn client(&self) -> reqwest::Client {
        self.client.clone()
    }

    pub fn upload_client(&self) -> reqwest::Client {
        self.upload_client.clone()
    }

    /// Create (upsert) a run with an explicit entity/project/name and return the
    /// resolved `(entity, project, name)` from the server.
    pub async fn upsert_run(
        &self,
        entity: String,
        project: String,
        name: String,
    ) -> Result<(String, String, String), ApiError> {
        let variables = upsert_bucket::Variables {
            entity: Some(entity),
            name: Some(name),
            project: Some(project),
            id: None,
            commit: None,
            config: None,
            debug: None,
            description: None,
            display_name: None,
            group_name: None,
            host: None,
            job_type: None,
            notes: None,
            program: None,
            repo: None,
            state: None,
            summary_metrics: None,
            sweep: None,
            tags: None,
        };
        let request_body = UpsertBucket::build_query(variables);

        let mut res: graphql_client::Response<upsert_bucket::ResponseData> = self
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

        let bucket = res
            .data
            .ok_or_else(|| ApiError::NoResponse("UpsertBucket query returned empty data".into()))?
            .upsert_bucket
            .ok_or_else(|| {
                ApiError::NoResponse(
                    "UpsertBucket query returned data with no upsert_bucket in response".into(),
                )
            })?
            .bucket
            .ok_or_else(|| {
                ApiError::NoResponse(
                    "UpsertBucket query returned data with no bucket in upsert_bucket".into(),
                )
            })?;
        let project = bucket.project.ok_or_else(|| {
            ApiError::NoResponse("UpsertBucket query returned data with no project in bucket".into())
        })?;

        Ok((project.entity.name, project.name, bucket.name))
    }
}
