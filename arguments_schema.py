from dataclasses import dataclass, field
from typing import Optional

import transformers

@dataclass
class ModelArguments:
    base_model: str
    lora_adapter_dir: Optional[str] = field(default=None)
    # quantization hyperparams
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use."}
    )
    # lora hyperparams
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DatasetArguments:
    prompt_template: str
    data_path: Optional[str] = field(default=None)


@dataclass
class InferenceArguments: 
    output_file: str
    batch_size: Optional[int] = field(default=1)
