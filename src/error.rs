use std::fmt::Display;
use std::future::Future;

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// A custom error type that combines a Reqwest error with the response body.
///
/// Ported from the `wandb-rs` baseline: wraps a [`reqwest::Error`] together with
/// the response body as a string, which helps debugging failed HTTP requests.
#[derive(Debug)]
pub struct ReqwestErrorWithBody {
    error: reqwest::Error,
    body: Result<String, reqwest::Error>,
}

impl Display for ReqwestErrorWithBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Request error:")?;
        writeln!(f, "{}", self.error)?;
        match &self.body {
            Ok(body) => {
                writeln!(f, "Response body:")?;
                writeln!(f, "{body}")?;
            }
            Err(err) => {
                writeln!(f, "Failed to fetch body:")?;
                writeln!(f, "{err}")?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for ReqwestErrorWithBody {}

pub trait ReqwestBadResponse {
    fn maybe_err(self) -> impl Future<Output = Result<Self, ReqwestErrorWithBody>>
    where
        Self: Sized;
}

impl ReqwestBadResponse for reqwest::Response {
    async fn maybe_err(self) -> Result<Self, ReqwestErrorWithBody>
    where
        Self: Sized,
    {
        let error = self.error_for_status_ref();
        if let Err(error) = error {
            let body = self.text().await;
            Err(ReqwestErrorWithBody { body, error })
        } else {
            Ok(self)
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ApiError {
    #[error("api request failed: {0}")]
    RequestErrorWithBody(#[from] ReqwestErrorWithBody),

    #[error("api request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("graphql query failed: {0:?}")]
    QueryFailed(Vec<graphql_client::Error>),

    #[error("serialize data to json failed: {0}")]
    SerializeJson(#[from] serde_json::Error),

    #[error("no response from query: {0}")]
    NoResponse(String),

    #[error("could not resolve a wandb API key: {0}")]
    MissingApiKey(String),
}

impl From<ApiError> for PyErr {
    fn from(err: ApiError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}
