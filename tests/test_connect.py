import asyncio

import pytest
import wandb_ft


def test_exports():
    assert set(wandb_ft.__all__) == {"Api", "Html", "Image", "Run", "Video", "connect"}


def test_connect_missing_key(tmp_path, monkeypatch):
    """connect() raises when no API key is configured in netrc."""
    empty = tmp_path / "netrc"
    empty.write_text("")
    monkeypatch.setenv("NETRC", str(empty))

    async def go():
        await wandb_ft.connect()

    with pytest.raises(RuntimeError):
        asyncio.run(go())


def test_connect_succeeds_with_real_netrc():
    """With the configured netrc, connect() resolves to an Api object."""

    async def go():
        api = await wandb_ft.connect()
        assert isinstance(api, wandb_ft.Api)

    asyncio.run(go())
