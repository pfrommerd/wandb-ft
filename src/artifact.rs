use std::path::PathBuf;

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[derive(Clone)]
pub enum ArtifactEntrySource {
    File(PathBuf),
    Bytes(Vec<u8>),
}

#[derive(Clone)]
pub struct ArtifactEntry {
    pub name: String,
    pub source: ArtifactEntrySource,
}

#[derive(Clone)]
pub struct ArtifactSnapshot {
    pub name: String,
    pub type_name: String,
    pub description: Option<String>,
    pub entries: Vec<ArtifactEntry>,
}

/// A wandb artifact containing files or in-memory byte entries.
#[pyclass]
pub struct Artifact {
    name: String,
    type_name: String,
    description: Option<String>,
    entries: Vec<ArtifactEntry>,
}

#[pymethods]
impl Artifact {
    #[new]
    #[pyo3(signature = (name, type_name, description=None))]
    fn new(name: String, type_name: String, description: Option<String>) -> PyResult<Self> {
        if name.is_empty() {
            return Err(PyValueError::new_err("Artifact name must not be empty"));
        }
        if type_name.is_empty() {
            return Err(PyValueError::new_err("Artifact type must not be empty"));
        }
        Ok(Self {
            name,
            type_name,
            description,
            entries: Vec::new(),
        })
    }

    /// Add a local file to the artifact. `name` defaults to the source filename.
    #[pyo3(signature = (path, name=None))]
    fn add_file(&mut self, path: String, name: Option<String>) -> PyResult<()> {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(PyFileNotFoundError::new_err(format!(
                "artifact file does not exist: {}",
                path.display()
            )));
        }
        let name = match name {
            Some(name) => name,
            None => path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| PyValueError::new_err("artifact file path has no filename"))?
                .to_string(),
        };
        self.add_entry(name, ArtifactEntrySource::File(path))
    }

    /// Add an in-memory bytes object to the artifact under `name`.
    fn add_bytes(&mut self, data: &Bound<'_, PyAny>, name: String) -> PyResult<()> {
        let bytes = data.cast::<PyBytes>().map_err(|_| {
            PyValueError::new_err("Artifact.add_bytes(data, name) expects a bytes object")
        })?;
        self.add_entry(name, ArtifactEntrySource::Bytes(bytes.as_bytes().to_vec()))
    }
}

impl Artifact {
    pub fn snapshot(&self) -> PyResult<ArtifactSnapshot> {
        if self.entries.is_empty() {
            return Err(PyValueError::new_err(
                "cannot log an Artifact with no files; call add_file or add_bytes first",
            ));
        }
        Ok(ArtifactSnapshot {
            name: self.name.clone(),
            type_name: self.type_name.clone(),
            description: self.description.clone(),
            entries: self.entries.clone(),
        })
    }

    fn add_entry(&mut self, name: String, source: ArtifactEntrySource) -> PyResult<()> {
        if name.is_empty() {
            return Err(PyValueError::new_err(
                "artifact entry name must not be empty",
            ));
        }
        if name.starts_with('/') || name.contains("..") {
            return Err(PyValueError::new_err(
                "artifact entry name must be a relative path without '..' components",
            ));
        }
        self.entries.push(ArtifactEntry { name, source });
        Ok(())
    }
}
