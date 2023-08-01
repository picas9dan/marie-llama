from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_path: str


@dataclass
class DatasetArguments:
    data_path: Optional[str] = field(default=None)
