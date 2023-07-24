import os

from peft import (
    get_peft_model, 
    LoraConfig, 
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer
import torch

from arguments_schema import ModelArgs


def get_model(model_args: ModelArgs, is_train: bool):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_args.bits == 4,
        load_in_8bit=model_args.bits == 8,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_args.base_model,
        device_map={"":0},
        quantization_config=bnb_config,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN")
    )

    if is_train:
        model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()
    
    if model_args.lora_adapter_dir is not None:
        print("Loading adapters from disk.")
        model = PeftModel.from_pretrained(model, model_args.lora_adapter_dir, is_trainable=True)
    else:
        print("Adding LoRA modules.")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not is_train,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, config)

    if is_train:
        model.config.use_cache = False
        model.print_trainable_parameters()
        
    print("Loaded model.")

    return model


def add_pad_token(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    assert len(tokenizer) == model.vocab_size
    assert tokenizer.add_special_tokens(dict(pad_token="<pad>")) == 1

    # Update model's embeddings data to include pad_token
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings_data = model.get_input_embeddings().weight.data
    output_embeddings_data = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings_data[:-1].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings_data[:-1].mean(dim=0, keepdim=True)

    input_embeddings_data[-1:] = input_embeddings_avg
    output_embeddings_data[-1:] = output_embeddings_avg

    assert len(tokenizer) == model.vocab_size


def get_model_and_tokenizer(model_args: ModelArgs, is_train: bool):
    model = get_model(model_args, is_train)
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.base_model,
        padding_side="right",
        use_fast=False,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN")
    )
    add_pad_token(model, tokenizer)

    return model, tokenizer
