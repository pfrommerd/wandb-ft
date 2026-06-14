import base64

import numpy as np
import pytest
import wandb_ft

# 1x1 transparent PNG.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def test_html_constructs_from_string():
    media = wandb_ft.Html("<h1>Hello</h1>")

    assert isinstance(media, wandb_ft.Html)


def test_image_constructs_from_encoded_png_bytes():
    media = wandb_ft.Image(PNG_BYTES)

    assert isinstance(media, wandb_ft.Image)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3, 1), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((2, 3, 4), dtype=np.uint8),
        np.zeros((2, 3, 3), dtype=np.float32),
        np.zeros((2, 3, 3), dtype=np.float64),
    ],
)
def test_image_constructs_from_supported_numpy_arrays(array):
    media = wandb_ft.Image(array)

    assert isinstance(media, wandb_ft.Image)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((2,), dtype=np.uint8), r"2D \(H, W\) or 3D \(H, W, C\)"),
        (np.zeros((2, 3, 2), dtype=np.uint8), r"channels must be 1 .* 3 .* 4"),
        (np.zeros((2, 3, 3), dtype=np.int32), "unsupported array dtype"),
    ],
)
def test_image_rejects_unsupported_numpy_arrays(array, message):
    with pytest.raises((TypeError, ValueError), match=message):
        wandb_ft.Image(array)


def test_image_rejects_non_image_bytes():
    with pytest.raises(ValueError, match="could not detect image format"):
        wandb_ft.Image(b"not an image")


def test_video_constructs_from_raw_bytes():
    media = wandb_ft.Video(b"raw encoded mp4 bytes")

    assert isinstance(media, wandb_ft.Video)


def test_video_rejects_non_array_non_bytes():
    with pytest.raises(TypeError, match="expects raw video bytes or a numpy array"):
        wandb_ft.Video("not bytes")


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 3, 4), dtype=np.uint8),
        np.zeros((2, 3, 4, 1), dtype=np.uint8),
        np.zeros((2, 3, 4, 4), dtype=np.uint8),
    ],
)
def test_video_rejects_invalid_array_shapes(array):
    with pytest.raises(ValueError, match=r"shape \(frames, height, width, 3\)"):
        wandb_ft.Video(array)
