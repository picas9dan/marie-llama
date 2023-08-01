from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_path: str


@dataclass
class DatasetArguments:
    data_path: str
    source_max_len: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=512,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )