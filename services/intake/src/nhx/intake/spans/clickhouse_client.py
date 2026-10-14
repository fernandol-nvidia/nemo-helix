# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse client for Intake spans."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from clickhouse_connect.driver.external import ExternalData
from fastapi import HTTPException, Request
from nhx.intake.config import IntakeConfig
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.spans.clickhouse_migrations import parse_clickhouse_url, run_clickhouse_migrations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClickHouseSettings:
    """Connection settings for Intake's ClickHouse storage."""

    url: str
    user: str
    password: str
    database: str

    @classmethod
    def from_config(cls, config: IntakeConfig | None = None) -> ClickHouseSettings:
        cfg = config or IntakeConfig()
        clickhouse_config = cfg.clickhouse_config
        return cls(
            url=clickhouse_config.url,
            user=clickhouse_config.user,
            password=clickhouse_config.password,
            database=clickhouse_config.database,
        )


class ClickHouseSpanClient:
    """Small async-friendly wrapper around clickhouse-connect."""

    def __init__(self, settings: ClickHouseSettings | None = None) -> None:
        self.settings = settings or ClickHouseSettings.from_config()
        # The native async driver binds each client to the event loop that opened it. The
        # platform test harness runs lifespan tasks and requests on different loops.
        self._clients: dict[asyncio.AbstractEventLoop, Any] = {}
        self._bootstrapped = False
        self._bootstrap_lock = asyncio.Lock()

    @property
    def database(self) -> str:
        return self.settings.database

    async def bootstrap_schema(self) -> None:
        async with self._bootstrap_lock:
            if self._bootstrapped:
                return
            await asyncio.to_thread(run_clickhouse_migrations, self.settings)
            self._bootstrapped = True

    async def command(
        self,
        query: str,
        *,
        parameters: Sequence[Any] | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Any:
        await self.bootstrap_schema()
        raw_client = await self._get_raw_client()
        return await raw_client.command(query, parameters=parameters, settings=settings)

    async def query(
        self,
        query: str,
        *,
        parameters: Sequence[Any] | dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
        external_data: ExternalData | None = None,
    ) -> Any:
        await self.bootstrap_schema()
        raw_client = await self._get_raw_client()
        return await raw_client.query(
            query,
            parameters=parameters,
            settings=settings,
            external_data=external_data,
        )

    async def insert(
        self,
        table: str,
        rows: Sequence[Sequence[Any]],
        *,
        column_names: Sequence[str],
    ) -> Any:
        if not rows:
            return None
        await self.bootstrap_schema()
        raw_client = await self._get_raw_client()
        return await raw_client.insert(table, rows, column_names=column_names, database=self.database)

    async def close(self) -> None:
        current_loop = asyncio.get_running_loop()
        clients = list(self._clients.items())
        self._clients.clear()
        for loop, client in clients:
            # A client whose loop is still running elsewhere can only be closed on that loop.
            if loop is current_loop or loop.is_closed():
                await _close_client(client)

    async def _get_raw_client(self) -> Any:
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            async with self._bootstrap_lock:
                client = self._clients.get(loop)
                if client is None:
                    client = await self._create_raw_client(database=self.database)
                    self._clients[loop] = client
        return client

    async def _create_raw_client(self, *, database: str) -> Any:
        clickhouse_connect = _import_clickhouse_connect()
        parsed = parse_clickhouse_url(self.settings.url)
        return await clickhouse_connect.get_async_client(
            host=parsed.host,
            port=parsed.port,
            secure=parsed.secure,
            username=self.settings.user,
            password=self.settings.password,
            database=database,
        )


async def bootstrap_schema(client: ClickHouseSpanClient) -> None:
    """Create the Intake ClickHouse schema if it does not exist."""

    await client.bootstrap_schema()


async def get_clickhouse_client(request: Request) -> AsyncIterator[ClickHouseSpanClient]:
    """FastAPI dependency for Intake's service-owned ClickHouse client."""

    # Platform-mounted app uses `intake_service`; standalone service app uses `service`.
    service = getattr(request.app.state, "intake_service", None)
    if service is None:
        service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Intake service is not attached to this application")

    service_client = getattr(service, "clickhouse_client", None)
    if service_client is None:
        raise HTTPException(
            status_code=503,
            detail="ClickHouse spans storage is disabled or was not initialized during Intake startup",
        )

    try:
        await bootstrap_schema(service_client)
    except Exception as exc:
        logger.exception("ClickHouse spans storage is unavailable")
        raise HTTPException(status_code=503, detail=CLICKHOUSE_UNAVAILABLE_MESSAGE) from exc
    yield service_client


def _import_clickhouse_connect() -> Any:
    import clickhouse_connect

    return clickhouse_connect


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        result = close()
        if hasattr(result, "__await__"):
            await result
