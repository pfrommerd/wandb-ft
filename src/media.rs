//! wandb media logging: `Image`, `Html`, `Video`, and plotly figures.
//!
//! Each media value is turned into a [`MediaFile`] (the encoded file bytes plus
//! the metadata wandb records). At log time the file is uploaded to the run and
//! a `{"_type": ...-file, "path", "sha256", "size", ...}` record is inserted into
//! the logged row, matching the official wandb media format.

use std::collections::HashMap;
use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use sha2::{Digest, Sha256};

use crate::data_value::{DataValue, LogData, pyobj_to_data_value};
use crate::video;

/// An encoded media file plus the extra wandb metadata for its `_type` record.
#[derive(Clone)]
pub struct MediaFile {
    bytes: Vec<u8>,
    subdir: &'static str,
    ext: &'static str,
    log_type: &'static str,
    extra: Vec<(String, DataValue)>,
}

impl MediaFile {
    /// Build the run-relative upload path and the history `_type` record for this
    /// file, given the metric key and step it is logged under.
    pub fn record(&self, key: &str, step: u64) -> (String, DataValue) {
        let hex = sha256_hex(&self.bytes);
        let id = &hex[..20];
        let path = format!(
            "{}/{}_{}_{}{}",
            self.subdir,
            sanitize_key(key),
            step,
            id,
            self.ext
        );

        let mut map: HashMap<String, DataValue> = HashMap::new();
        map.insert("_type".into(), DataValue::String(self.log_type.into()));
        map.insert("path".into(), DataValue::String(path.clone()));
        map.insert("sha256".into(), DataValue::String(hex.clone()));
        map.insert("size".into(), DataValue::Int(self.bytes.len() as u64));
        for (k, v) in &self.extra {
            map.insert(k.clone(), v.clone());
        }
        (path, DataValue::Dict(map))
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A parsed `Run.log` row: scalar metrics plus any media that must be uploaded.
pub struct ParsedRow {
    pub scalars: LogData,
    pub media: Vec<(String, MediaFile)>,
}

/// Parse a Python `dict` into scalar metrics and pending media uploads.
pub fn parse_row(dict: &Bound<'_, PyDict>) -> PyResult<ParsedRow> {
    let mut scalars = LogData::new();
    let mut media = Vec::new();
    for (key, value) in dict.iter() {
        let key: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("Run.log metric keys must be strings"))?;
        if let Some(m) = try_extract_media(&value)? {
            media.push((key, m));
        } else {
            scalars.insert(key, pyobj_to_data_value(&value)?);
        }
    }
    Ok(ParsedRow { scalars, media })
}

/// Recognize a media value: one of our media classes, or a plotly figure.
fn try_extract_media(value: &Bound<'_, PyAny>) -> PyResult<Option<MediaFile>> {
    if let Ok(m) = value.cast::<Image>() {
        return Ok(Some(m.borrow().file.clone()));
    }
    if let Ok(m) = value.cast::<Video>() {
        return Ok(Some(m.borrow().file.clone()));
    }
    if let Ok(m) = value.cast::<Html>() {
        return Ok(Some(m.borrow().file.clone()));
    }
    try_plotly(value)
}

// ---------------------------------------------------------------------------
// Python-facing media classes
// ---------------------------------------------------------------------------

/// An HTML snippet to log. `data` is the HTML string.
#[pyclass]
pub struct Html {
    file: MediaFile,
}

#[pymethods]
impl Html {
    #[new]
    #[pyo3(signature = (data))]
    fn new(data: &str) -> Self {
        Html {
            file: MediaFile {
                bytes: inject_head(data).into_bytes(),
                subdir: "media/html",
                ext: ".html",
                log_type: "html-file",
                extra: Vec::new(),
            },
        }
    }
}

/// An image to log. `data` is either raw encoded image bytes (PNG/JPEG) or a
/// numpy array of shape `(H, W)`, `(H, W, 3)`, or `(H, W, 4)` (uint8 or float
/// in `[0, 1]`).
#[pyclass]
pub struct Image {
    file: MediaFile,
}

#[pymethods]
impl Image {
    #[new]
    #[pyo3(signature = (data))]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Image {
            file: build_image(data)?,
        })
    }
}

/// A video to log. `data` is either raw encoded video bytes (treated as MP4) or
/// a numpy array of shape `(frames, H, W, 3)` (uint8 or float in `[0, 1]`),
/// which is stream-encoded to H.264 MP4 via the `ffmpeg` CLI at `fps`.
#[pyclass]
pub struct Video {
    file: MediaFile,
}

#[pymethods]
impl Video {
    #[new]
    #[pyo3(signature = (data, fps=4))]
    fn new(data: &Bound<'_, PyAny>, fps: u32) -> PyResult<Self> {
        Ok(Video {
            file: build_video(data, fps)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

fn build_image(data: &Bound<'_, PyAny>) -> PyResult<MediaFile> {
    if let Ok(b) = data.cast::<PyBytes>() {
        return image_from_bytes(b.as_bytes());
    }
    let shape: Vec<usize> = data
        .getattr("shape")
        .and_then(|s| s.extract())
        .map_err(|_| {
            PyTypeError::new_err("Image(data=...) expects raw image bytes or a numpy array")
        })?;
    image_from_array(data, &shape)
}

fn image_from_bytes(bytes: &[u8]) -> PyResult<MediaFile> {
    let format = image::guess_format(bytes)
        .map_err(|e| PyValueError::new_err(format!("could not detect image format: {e}")))?;
    let img = image::load_from_memory_with_format(bytes, format)
        .map_err(|e| PyValueError::new_err(format!("could not decode image: {e}")))?;
    let (format_name, ext) = match format {
        ImageFormat::Jpeg => ("jpeg", ".jpg"),
        _ => ("png", ".png"),
    };
    Ok(MediaFile {
        bytes: bytes.to_vec(),
        subdir: "media/images",
        ext,
        log_type: "image-file",
        extra: vec![
            ("format".into(), DataValue::String(format_name.into())),
            ("width".into(), DataValue::Int(img.width() as u64)),
            ("height".into(), DataValue::Int(img.height() as u64)),
        ],
    })
}

fn image_from_array(data: &Bound<'_, PyAny>, shape: &[usize]) -> PyResult<MediaFile> {
    let (h, w, c) = match shape.len() {
        2 => (shape[0], shape[1], 1usize),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(PyValueError::new_err(
                "Image numpy array must be 2D (H, W) or 3D (H, W, C)",
            ));
        }
    };
    if !(c == 1 || c == 3 || c == 4) {
        return Err(PyValueError::new_err(
            "Image numpy array channels must be 1 (gray), 3 (RGB), or 4 (RGBA)",
        ));
    }

    let buf = array_to_u8(data)?;
    if buf.len() != h * w * c {
        return Err(PyValueError::new_err(
            "Image array data did not match its shape",
        ));
    }
    let (w32, h32) = (w as u32, h as u32);
    let dynimg = match c {
        1 => DynamicImage::ImageLuma8(
            ImageBuffer::from_raw(w32, h32, buf)
                .ok_or_else(|| PyValueError::new_err("invalid grayscale image buffer"))?,
        ),
        3 => DynamicImage::ImageRgb8(
            ImageBuffer::from_raw(w32, h32, buf)
                .ok_or_else(|| PyValueError::new_err("invalid RGB image buffer"))?,
        ),
        _ => DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(w32, h32, buf)
                .ok_or_else(|| PyValueError::new_err("invalid RGBA image buffer"))?,
        ),
    };

    let mut out = Cursor::new(Vec::new());
    dynimg
        .write_to(&mut out, ImageFormat::Png)
        .map_err(|e| PyValueError::new_err(format!("could not encode PNG: {e}")))?;

    Ok(MediaFile {
        bytes: out.into_inner(),
        subdir: "media/images",
        ext: ".png",
        log_type: "image-file",
        extra: vec![
            ("format".into(), DataValue::String("png".into())),
            ("width".into(), DataValue::Int(w as u64)),
            ("height".into(), DataValue::Int(h as u64)),
        ],
    })
}

fn build_video(data: &Bound<'_, PyAny>, fps: u32) -> PyResult<MediaFile> {
    if let Ok(b) = data.cast::<PyBytes>() {
        return Ok(MediaFile {
            bytes: b.as_bytes().to_vec(),
            subdir: "media/videos",
            ext: ".mp4",
            log_type: "video-file",
            extra: Vec::new(),
        });
    }
    let shape: Vec<usize> = data.getattr("shape").and_then(|s| s.extract()).map_err(|_| {
        PyTypeError::new_err(
            "Video(data=...) expects raw video bytes or a numpy array of shape (frames, H, W, 3)",
        )
    })?;
    if shape.len() != 4 || shape[3] != 3 {
        return Err(PyValueError::new_err(
            "Video numpy array must have shape (frames, height, width, 3)",
        ));
    }
    let (t, h, w) = (shape[0], shape[1], shape[2]);
    let buf = array_to_u8(data)?;
    let mp4 = video::encode_h264_mp4(&buf, w as u32, h as u32, t, fps)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(MediaFile {
        bytes: mp4,
        subdir: "media/videos",
        ext: ".mp4",
        log_type: "video-file",
        extra: vec![
            ("width".into(), DataValue::Int(w as u64)),
            ("height".into(), DataValue::Int(h as u64)),
        ],
    })
}

fn try_plotly(value: &Bound<'_, PyAny>) -> PyResult<Option<MediaFile>> {
    let module: String = value
        .get_type()
        .getattr("__module__")
        .and_then(|m| m.extract())
        .unwrap_or_default();
    if module.starts_with("plotly") && value.hasattr("to_plotly_json")? {
        let json: String = value.call_method0("to_json")?.extract()?;
        return Ok(Some(MediaFile {
            bytes: json.into_bytes(),
            subdir: "media/plotly",
            ext: ".plotly.json",
            log_type: "plotly-file",
            extra: Vec::new(),
        }));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read any uint8/float numpy array into a flat `Vec<u8>` in C (row-major) order.
/// Floats are assumed to be in `[0, 1]` and scaled to `[0, 255]`.
fn array_to_u8(arr: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(a) = arr.cast::<PyArrayDyn<u8>>() {
        return Ok(a.readonly().as_array().iter().copied().collect());
    }
    if let Ok(a) = arr.cast::<PyArrayDyn<f32>>() {
        return Ok(a
            .readonly()
            .as_array()
            .iter()
            .map(|&x| float_to_u8(x as f64))
            .collect());
    }
    if let Ok(a) = arr.cast::<PyArrayDyn<f64>>() {
        return Ok(a
            .readonly()
            .as_array()
            .iter()
            .map(|&x| float_to_u8(x))
            .collect());
    }
    Err(PyTypeError::new_err(
        "unsupported array dtype for media; expected uint8 or float32/float64",
    ))
}

fn float_to_u8(x: f64) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Sanitize a metric key for use in a filename (wandb does the same).
fn sanitize_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Inject the wandb stylesheet/base tag into an HTML snippet (matching the
/// official client's default `inject=True` behavior).
fn inject_head(html: &str) -> String {
    const SNIPPET: &str = "<base target=\"_blank\"><link rel=\"stylesheet\" type=\"text/css\" \
         href=\"https://app.wandb.ai/normalize.css\" />";
    if let Some(pos) = html.to_lowercase().find("<head>") {
        let at = pos + "<head>".len();
        format!("{}{}{}", &html[..at], SNIPPET, &html[at..])
    } else {
        format!("<head>{SNIPPET}</head>{html}")
    }
}
