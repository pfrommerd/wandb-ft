use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

mod artifact;
mod backend;
mod data_value;
mod error;
mod gql;
mod media;
mod netrc;
mod run;
mod video;

use artifact::Artifact;
use backend::Backend;
use run::RunState;

/// Connection to the wandb backend. Created via [`connect`].
#[pyclass]
struct Api {
    inner: Arc<Backend>,
}

#[pymethods]
impl Api {
    /// Create (upsert) a run with an explicit entity/project/name.
    #[pyo3(signature = (entity, project, name, config=None, summary=None))]
    fn create_run<'py>(
        &self,
        py: Python<'py>,
        entity: String,
        project: String,
        name: String,
        config: Option<&Bound<'py, PyAny>>,
        summary: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let config = json_dumps(py, config)?;
        let summary = json_dumps(py, summary)?;
        let backend = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (entity, project, name) = backend
                .upsert_run(entity, project, name, config, summary)
                .await?;
            let state = RunState::new(
                backend.client(),
                backend.upload_client(),
                backend.base_url().to_string(),
                entity,
                project,
                name,
            );
            Ok(Run {
                state: Arc::new(state),
            })
        })
    }
}

/// A single wandb run. Created via [`Api::create_run`].
#[pyclass]
struct Run {
    state: Arc<RunState>,
}

#[pymethods]
impl Run {
    /// Log a row of metrics at the given step.
    #[pyo3(signature = (metrics, step))]
    fn log<'py>(
        &self,
        py: Python<'py>,
        metrics: &Bound<'py, PyDict>,
        step: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Parse the dict (encoding media, reading numpy) synchronously while
        // holding the GIL, so the resulting Send value can cross the await
        // boundary; media uploads then happen inside the async block.
        let row = media::parse_row(metrics)?;
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            state.log(row, step).await?;
            Ok(())
        })
    }

    /// Update the run summary with a JSON-serializable object.
    fn update_summary<'py>(
        &self,
        py: Python<'py>,
        summary: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let summary = json_dumps(py, Some(summary))?.unwrap_or_else(|| "{}".into());
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            state.update_summary(summary).await?;
            Ok(())
        })
    }

    /// Log an artifact and upload all of its files.
    #[pyo3(signature = (artifact, aliases=None))]
    fn log_artifact<'py>(
        &self,
        py: Python<'py>,
        artifact: &Bound<'py, Artifact>,
        aliases: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let artifact = artifact.borrow().snapshot()?;
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            state.log_artifact(artifact, aliases).await?;
            Ok(())
        })
    }

    /// Flush and mark the run as finished on the backend.
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let state = self.state.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            state.finish().await?;
            Ok(())
        })
    }
}

/// Connect to wandb, reading the API key from netrc. Raises if no key is configured.
#[pyfunction]
fn connect(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let api_key = netrc::read_api_key()?;
        Ok(Api {
            inner: Arc::new(Backend::new(&api_key)),
        })
    })
}

#[pymodule(gil_used = false)]
fn _wandb_ft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connect, m)?)?;
    m.add_class::<Api>()?;
    m.add_class::<Run>()?;
    m.add_class::<media::Html>()?;
    m.add_class::<media::Image>()?;
    m.add_class::<media::Video>()?;
    m.add_class::<Artifact>()?;
    Ok(())
}

fn json_dumps(py: Python<'_>, value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let json = PyModule::import(py, "json")?;
    json.call_method1("dumps", (value,))
        .and_then(|s| s.extract())
        .map(Some)
}
