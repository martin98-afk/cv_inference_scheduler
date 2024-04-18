from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from typing import List


class ModelStatus(Enum):
    BOOTING = "booting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class ModelInstance:
    model_name: str
    model_path: str
    model_task: str
    model_config: str
    port: int
    process: Process
    device: str = "cpu"
    pid_list: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.BOOTING


@dataclass
class StreamInstance:
    pid: str
    process: Process