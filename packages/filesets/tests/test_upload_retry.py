# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File uploads rebuild consumed bodies without replacing the client's retry policy."""

from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from filesets.filesystem.filesystem import AsyncFilesetFileSystem, FilesetFileSystem
from fsspec.callbacks import Callback
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient

POLICY = replace(PLATFORM_DEFAULT_RETRY_POLICY, backoff_base=0)


@pytest.mark.parametrize("status,failures,attempts", [(429, 1, 2), (500, 1, 2), (503, 4, 3), (403, 1, 1)])
def test_sync_upload_reopens_file(tmp_path: Path, status: int, failures: int, attempts: int) -> None:
    payload = b"first chunk\nsecond chunk\n" * 100
    path = tmp_path / "upload.bin"
    path.write_bytes(payload)
    bodies: list[bytes] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        assert int(request.headers["Content-Length"]) == len(payload)
        return httpx.Response(
            status if len(bodies) <= failures else 200,
            json={
                "file_ref": "default/test#upload.bin",
                "file_url": str(request.url),
                "path": "upload.bin",
                "size": len(payload),
            },
        )

    callback = Callback()
    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = FilesClient(base_url="http://test", workspace="default", http_client=http, retry=POLICY)
        fs = FilesetFileSystem(client=client, skip_instance_cache=True)
        fs.blocksize = 17
        if attempts <= failures:
            with pytest.raises(NemoHTTPError):
                fs.put_file(str(path), "default/test/upload.bin", callback=callback)
        else:
            fs.put_file(str(path), "default/test/upload.bin", callback=callback)
    assert bodies == [payload] * attempts
    assert callback.value == callback.size == len(payload)


@pytest.mark.parametrize("status,failures,attempts", [(429, 1, 2), (500, 1, 2), (503, 4, 3), (403, 1, 1)])
async def test_async_upload_reopens_file(tmp_path: Path, status: int, failures: int, attempts: int) -> None:
    payload = b"first chunk\nsecond chunk\n" * 100
    path = tmp_path / "upload.bin"
    path.write_bytes(payload)
    bodies: list[bytes] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(await request.aread())
        assert int(request.headers["Content-Length"]) == len(payload)
        return httpx.Response(
            status if len(bodies) <= failures else 200,
            json={
                "file_ref": "default/test#upload.bin",
                "file_url": str(request.url),
                "path": "upload.bin",
                "size": len(payload),
            },
        )

    callback = Callback()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = AsyncFilesClient(base_url="http://test", workspace="default", http_client=http, retry=POLICY)
        fs = AsyncFilesetFileSystem(client=client, skip_instance_cache=True)
        fs.blocksize = 17
        if attempts <= failures:
            with pytest.raises(NemoHTTPError):
                await fs._put_file(str(path), "default/test/upload.bin", callback=callback)
        else:
            await fs._put_file(str(path), "default/test/upload.bin", callback=callback)
    assert bodies == [payload] * attempts
    assert callback.value == callback.size == len(payload)
