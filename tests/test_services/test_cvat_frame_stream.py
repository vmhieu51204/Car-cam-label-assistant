from __future__ import annotations

import asyncio

import httpx
import pytest

from src.services.cvat import CvatReadClient, CvatReadError


def test_parallel_cvat_reads_keep_frame_order_and_bound_concurrency():
    concurrent = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal concurrent, peak
        index = int(request.url.params["number"])
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep((7 - index % 8) * 0.001)
        concurrent -= 1
        return httpx.Response(200, content=str(index).encode())

    client = CvatReadClient(base_url="http://cvat.test", token="test", transport=httpx.MockTransport(handler))

    async def read() -> list[bytes]:
        return [frame async for frame in client.iter_job_frames(3, 19, quality="compressed")]

    assert asyncio.run(read()) == [str(index).encode() for index in range(19)]
    assert 1 < peak <= 8


def test_parallel_cvat_reads_report_failed_frame_index():
    async def handler(request: httpx.Request) -> httpx.Response:
        index = int(request.url.params["number"])
        return httpx.Response(503 if index == 6 else 200, content=b"frame")

    client = CvatReadClient(base_url="http://cvat.test", token="test", transport=httpx.MockTransport(handler))

    async def read() -> None:
        async for _ in client.iter_selected_job_frames(3, [4, 5, 6, 7]):
            pass

    with pytest.raises(CvatReadError, match="frame 6"):
        asyncio.run(read())
