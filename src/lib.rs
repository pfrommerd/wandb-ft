use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod backend;
mod data_value;
mod error;
mod gql;
mod media;
mod netrc;
mod run;
mod video;

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
    fn create_run<'py>(
        &self,
        py: Python<'py>,
        entity: String,
        project: String,
        name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let backend = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (entity, project, name) = backend.upsert_run(entity, project, name).await?;
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
    Ok(())
}
