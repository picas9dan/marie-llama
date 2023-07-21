from dataclasses import dataclass, field

import transformers

@dataclass
class ModelArgs:
    base_model: str


@dataclass
class DataArgs:
    train_data_path: str
    eval_data_path: str
    prompt_template: str
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
    batch_size: int = field(default=128)
    # quantization hyperparams
    bits: int = field(
        default=8,
        metadata={"help": "How many bits to use."}
    )
    # lora hyperparams
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=32)
    lora_dropout: float = field(default=0.05)
