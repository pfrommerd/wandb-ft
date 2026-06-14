//! Encode raw RGB frames to an H.264 MP4 by shelling out to the `ffmpeg` CLI.
//!
//! We intentionally avoid linking ffmpeg libraries; the `ffmpeg` binary is only
//! required when a `Video` is constructed from a numpy array (raw-bytes videos
//! need no re-encoding).

use std::io::Write;
use std::process::{Command, Stdio};

#[derive(thiserror::Error, Debug)]
pub enum VideoError {
    #[error(
        "the `ffmpeg` command-line tool was not found on PATH; it is required to \
         encode a Video from a numpy array. Install ffmpeg or pass raw encoded \
         video bytes instead."
    )]
    FfmpegNotFound,
    #[error("ffmpeg failed to encode the video:\n{0}")]
    EncodeFailed(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

/// Stream-encode contiguous RGB24 frames (row-major, `frames` images of
/// `width * height * 3` bytes) into an H.264 MP4 and return the bytes.
///
/// The raw frames are piped to ffmpeg via stdin; the MP4 is written to a temp
/// file (mp4 needs a seekable sink for `+faststart`) and read back.
pub fn encode_h264_mp4(
    rgb: &[u8],
    width: u32,
    height: u32,
    frames: usize,
    fps: u32,
) -> Result<Vec<u8>, VideoError> {
    let fps = fps.max(1);
    let frame_bytes = (width as usize) * (height as usize) * 3;
    if rgb.len() != frame_bytes * frames {
        return Err(VideoError::Other(format!(
            "frame buffer is {} bytes, expected {} ({frames} frames of {width}x{height} RGB)",
            rgb.len(),
            frame_bytes * frames
        )));
    }

    let dir = tempfile::tempdir()?;
    let output = dir.path().join("out.mp4");

    let mut child = match Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &fps.to_string(),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&output)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(VideoError::FfmpegNotFound);
        }
        Err(e) => return Err(e.into()),
    };

    // Write all frames to stdin, then close it so ffmpeg can finish.
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| VideoError::Other("failed to open ffmpeg stdin".into()))?;
        stdin.write_all(rgb)?;
    }

    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(VideoError::EncodeFailed(
            String::from_utf8_lossy(&out.stderr).trim().to_string(),
        ));
    }

    Ok(std::fs::read(&output)?)
}
