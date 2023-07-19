from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Optional
import torch

import transformers

@dataclass
class ModelArgs:
    base_model: str


@dataclass
class DataArgs:
    eval_dataset_size: int = field(default=1024)
    source_max_len: int = field(
        default=1024, # should be able to accommodate the longest input question
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    target_max_len: int = field(
        default=256, # should be able to accommodate the longest SPARQL query 
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        metadata={"help": "Dataset to finetune on."}
    )
    dataset_format: str = field(
        metadata={"help": "Dataset format used. [alpaca|self-instruct]."}
    )

@dataclass
class TrainArgs(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    batch_size: int = field(default=128)
    learning_rate: float = field(default=0.0002)
    num_epochs: int
    # quantization hyperparams
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use."}
    )
    # lora hyperparams
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=32)
    lora_dropout: float = field(default=0.05)
