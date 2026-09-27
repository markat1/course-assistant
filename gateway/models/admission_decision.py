from typing import Literal
from pydantic import BaseModel, Field
from gateway.models.workers.worker_state import WorkerState

class AdmissionDecision(BaseModel):
    status_code: Literal[200, 429, 503]
    reason: Literal["accepted", "no_eligible_workers", "kv_pressure"]
    workers: list[WorkerState] = Field(default_factory=list)