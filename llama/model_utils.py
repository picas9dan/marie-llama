import os

from peft import PeftModel
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
import torch

from llama.arguments_schema import ModelArguments


def get_model(model_args: ModelArguments):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=model_args.bits == 8,
        load_in_4bit=model_args.bits == 4,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LlamaForCausalLM.from_pretrained(
        model_args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    )

    if model_args.lora_adapter_dir:
        model = PeftModel.from_pretrained(model, model_args.lora_adapter_dir)

    return model


def get_tokenizer(base_model: str):
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    return tokenizer