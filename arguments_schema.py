from dataclasses import dataclass, field
from typing import Optional

import transformers

@dataclass
class ModelArgs:
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
class DataArgs:
    prompt_template: str
    train_data_path: Optional[str] = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    source_max_len: int = field(
        default=1024, # should be able to accommodate the longest input question
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    target_max_len: int = field(
        default=256, # should be able to accommodate the longest SPARQL query 
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class TrainArgs(transformers.Seq2SeqTrainingArguments):
    pass


@dataclass
class GenArgs:
    # Length arguments
    max_new_tokens: Optional[int] = field(default=512)
    min_new_tokens: Optional[int] = field(default=None)

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


@dataclass
class InferArgs: 
    output_file: str
    batch_size: Optional[int] = field(default=8)
