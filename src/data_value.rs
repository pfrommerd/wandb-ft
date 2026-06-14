use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
use serde::{Serialize, Serializer, ser::SerializeMap};

/// A single JSON-serializable value that can be logged to wandb.
///
/// Ported from the `wandb-rs` baseline, minus the `From<tuple>` ergonomics
/// (Python passes dicts, not Rust tuples).
#[derive(Debug, Clone, PartialEq)]
pub enum DataValue {
    Bool(bool),
    Int(u64),
    SignedInt(i64),
    Float(f64),
    String(String),
    #[allow(dead_code)]
    List(Vec<DataValue>),
    Dict(HashMap<String, DataValue>),
}

impl Serialize for DataValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DataValue::Bool(b) => serializer.serialize_bool(*b),
            DataValue::Int(i) => serializer.serialize_u64(*i),
            DataValue::SignedInt(i) => serializer.serialize_i64(*i),
            DataValue::Float(f) => serializer.serialize_f64(*f),
            DataValue::String(s) => serializer.serialize_str(s),
            DataValue::List(l) => l.serialize(serializer),
            DataValue::Dict(d) => {
                let mut map = serializer.serialize_map(Some(d.len()))?;
                for (k, v) in d {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

/// A row of metrics to be logged to wandb.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct LogData {
    data: HashMap<String, DataValue>,
}

impl LogData {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<DataValue>) {
        self.data.insert(key.into(), value.into());
    }

    /// Insert a value only if the key is not already present,
    /// allowing user-provided values to take precedence.
    pub fn insert_default(&mut self, key: impl Into<String>, value: impl Into<DataValue>) {
        self.data.entry(key.into()).or_insert_with(|| value.into());
    }
}

impl From<f64> for DataValue {
    fn from(v: f64) -> Self {
        DataValue::Float(v)
    }
}

impl Serialize for LogData {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.data.len()))?;
        for (k, v) in &self.data {
            map.serialize_entry(k, &v)?;
        }
        map.end()
    }
}

/// Convert a single Python metric value into a [`DataValue`].
///
/// Accepts Python `bool`/`int`/`float`/`str` and 0-d (scalar) numpy arrays /
/// numpy scalars. A numpy array with `ndim != 0` raises `ValueError`.
pub(crate) fn pyobj_to_data_value(value: &Bound<'_, PyAny>) -> PyResult<DataValue> {
    // `bool` must be checked before `int` (bool is a subclass of int in Python).
    if value.is_instance_of::<PyBool>() {
        return Ok(DataValue::Bool(value.extract::<bool>()?));
    }
    if value.is_instance_of::<PyInt>() {
        // Prefer signed to preserve negatives; fall back to u64 for very large.
        if let Ok(i) = value.extract::<i64>() {
            return Ok(DataValue::SignedInt(i));
        }
        return Ok(DataValue::Int(value.extract::<u64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(DataValue::Float(value.extract::<f64>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(DataValue::String(value.extract::<String>()?));
    }

    // numpy arrays / scalars: require a 0-d (scalar) shape.
    if let Ok(ndim_attr) = value.getattr("ndim") {
        if let Ok(ndim) = ndim_attr.extract::<usize>() {
            if ndim != 0 {
                return Err(PyValueError::new_err(format!(
                    "Run.log only accepts scalar values; got an array with ndim={ndim} \
                     (shape != ()). Reduce it to a scalar before logging."
                )));
            }
            // 0-d array / numpy scalar -> coerce to f64 via __float__.
            let f: f64 = value.call_method0("item")?.extract()?;
            return Ok(DataValue::Float(f));
        }
    }

    Err(PyTypeError::new_err(format!(
        "Unsupported metric value type: {}. Expected float, int, bool, str, or a \
         scalar numpy value.",
        value.get_type().name()?,
    )))
}
