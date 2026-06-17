"""RisingWaveMaterializationJob — the MaterializationJob the engine returns.

Mirrors FlinkMaterializationJob / LocalMaterializationJob. Retrieval (training/PIT) is
handled by ``RisingWaveOfflineStore`` — Feast routes get_historical_features to the
offline store, not the compute engine — so there is no retrieval job here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from feast.infra.common.materialization_job import (
    MaterializationJob,
    MaterializationJobStatus,
)


@dataclass
class RisingWaveMaterializationJob(MaterializationJob):
    def __init__(
        self,
        job_id: str,
        status: MaterializationJobStatus,
        error: Optional[BaseException] = None,
    ) -> None:
        super().__init__()
        self._job_id = job_id
        self._status = status
        self._error = error

    def status(self) -> MaterializationJobStatus:
        return self._status

    def error(self) -> Optional[BaseException]:
        return self._error

    def should_be_retried(self) -> bool:
        return False

    def job_id(self) -> str:
        return self._job_id

    def url(self) -> Optional[str]:
        return None
