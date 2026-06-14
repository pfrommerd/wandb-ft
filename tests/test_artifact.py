import pytest
import wandb_ft


def test_artifact_constructs_and_accepts_files_and_bytes(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"model")

    artifact = wandb_ft.Artifact("model", "model", description="checkpoint")
    artifact.add_file(str(path))
    artifact.add_file(str(path), name="nested/renamed.bin")
    artifact.add_bytes(b"metadata", "metadata.txt")

    assert isinstance(artifact, wandb_ft.Artifact)


@pytest.mark.parametrize(("name", "type_name"), [("", "dataset"), ("dataset", "")])
def test_artifact_rejects_empty_name_or_type(name, type_name):
    with pytest.raises(ValueError):
        wandb_ft.Artifact(name, type_name)


def test_artifact_rejects_missing_file(tmp_path):
    artifact = wandb_ft.Artifact("dataset", "dataset")

    with pytest.raises(FileNotFoundError):
        artifact.add_file(str(tmp_path / "missing.txt"))


@pytest.mark.parametrize("entry_name", ["", "/absolute.txt", "../escape.txt"])
def test_artifact_rejects_invalid_entry_names(tmp_path, entry_name):
    path = tmp_path / "data.txt"
    path.write_text("data")
    artifact = wandb_ft.Artifact("dataset", "dataset")

    with pytest.raises(ValueError):
        artifact.add_file(str(path), name=entry_name)


def test_artifact_add_bytes_requires_bytes():
    artifact = wandb_ft.Artifact("dataset", "dataset")

    with pytest.raises(ValueError, match="expects a bytes object"):
        artifact.add_bytes("not bytes", "data.txt")


def test_run_exposes_summary_and_artifact_methods():
    assert hasattr(wandb_ft.Run, "update_summary")
    assert hasattr(wandb_ft.Run, "log_artifact")
