import asyncio
import os
import sys
import tempfile

import numpy as np
import wandb_ft

"""
End-to-end integration test for wandb-ft.

This script tests:
- HTML logging
- Image logging
- Metric logging
- Configuration logging (on run creation)
- Summary updating
- Video logging
- Artifact creation/logging

Requirements:
- A W&B API key configured in ~/.netrc or via the NETRC environment variable.
- numpy installed.
- wandb-ft built/installed.

Usage:
    export WANDB_ENTITY="your-entity"
    export WANDB_PROJECT="your-project"
    python full_test.py
"""


async def main():
    # 1. Setup Entity and Project
    entity = os.getenv("WANDB_ENTITY", "dpfrommer-projects")
    project = os.getenv("WANDB_PROJECT", "wandb-ft")

    if not entity:
        print("Error: WANDB_ENTITY environment variable must be set.")
        print("Example: export WANDB_ENTITY='my-username'")
        sys.exit(1)

    # 2. Connect to wandb
    print(f"Connecting to W&B (Entity: {entity}, Project: {project})...")
    try:
        api = await wandb_ft.connect()
    except Exception as e:
        print(f"Failed to connect: {e}")
        print(
            "Make sure you have a valid API key in your ~/.netrc or set via NETRC env var."
        )
        sys.exit(1)

    # 3. Create Run (Tests Configuration and Initial Summary)
    print("Creating run...")
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "architecture": "Integration-Test-Model",
        "nested_config": {"optimizer": "Adam", "beta1": 0.9},
    }
    initial_summary = {"status": "starting"}

    run = await api.create_run(
        entity=entity,
        project=project,
        name="full-integration-test-run",
        config=config,
        summary=initial_summary,
    )
    print("Run created successfully.")

    # 4. Prepare Media Data
    print("Preparing media data...")
    # Image: 64x64 RGB random noise
    img_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image = wandb_ft.Image(img_data)

    # Video: 10 frames of 64x64 RGB random noise
    video_data = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
    video = wandb_ft.Video(video_data, fps=4)

    # HTML snippet
    html = wandb_ft.Html(
        "<h1>Integration Test</h1><p>This was logged from <b>wandb-ft</b>!</p>"
    )

    # 5. Log Metrics and Media
    print("Logging metrics, image, video, and HTML...")
    metrics = {
        "loss": 0.543,
        "accuracy": 0.88,
        "epoch": 1,
        "test_image": image,
        "test_video": video,
        "test_html": html,
    }
    await run.log(metrics, step=0)

    # 6. Update Summary
    print("Updating summary...")
    await run.update_summary(
        {"status": "completed", "final_accuracy": 0.92, "total_epochs": 1}
    )

    # 7. Create and Log Artifact
    print("Logging artifact...")
    # Create a temporary file to include in the artifact
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a dummy model weight file for integration testing.")
        temp_file_path = f.name

    try:
        artifact = wandb_ft.Artifact(
            name="test-model-artifact",
            type_name="model",
            description="An artifact containing dummy model weights",
        )
        # Add a file from disk
        artifact.add_file(temp_file_path, name="weights.txt")
        # Add bytes directly
        artifact.add_bytes(b"metadata: version=1.0, author=wandb-ft", "metadata.txt")

        await run.log_artifact(artifact, aliases=["latest", "test-pass"])
        print("Artifact logged successfully.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # 8. Finish the Run
    print("Finishing run...")
    await run.finish()
    print("\nIntegration test finished successfully!")
    print(
        f"You can view the run at: https://wandb.ai/{entity}/{project}/runs/full-integration-test-run"
    )

    print(
        f"You can view the run at: https://wandb.ai/{entity}/{project}/runs/full-integration-test-run"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
