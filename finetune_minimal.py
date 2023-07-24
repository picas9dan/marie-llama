import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


PROMPT_TEMPLATE = "{question}\n\n###\n\n{query}"

def construct_example(example):
    example["sentence"] = PROMPT_TEMPLATE.format(**example)
    return example


def main():
    base_model = "meta-llama/Llama-2-7b-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0

    data = Dataset.from_json("./data/train_full_20230721.json")
    data = data.map(construct_example)
    data = data.map(lambda samples: tokenizer(samples["sentence"]), batched=True)

    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="/rds/user/nmdt2/hpc-work/outputs/20230724_minimal",
            optim="paged_adamw_8bit"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    model.save_pretrained("/rds/user/nmdt2/hpc-work/outputs/20230724_minimal")

if __name__ == "__main__":
    main()
