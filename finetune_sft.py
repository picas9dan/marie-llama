import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, TrainerCallback, TrainingArguments
from peft import LoraConfig
from datasets import Dataset
from trl import SFTTrainer


PROMPT_TEMPLATE = "{question}\n\n###\n\n{query}"

def construct_examples(examples):
    output_texts = []

    for i in range(len(examples["question"])):
        example = {k: v[i] for k, v in examples.items()}
        text = PROMPT_TEMPLATE.format(**example)
        output_texts.append(text)

    return output_texts


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def finetune():
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
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    dataset = Dataset.from_json("./data/train_full_20230721.json")
    callbacks = [PeftSavingCallback]

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=construct_examples,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            warmup_steps=2,
            learning_rate=2e-4,
            bf16=True,
            save_steps=10,
            logging_steps=1,
            output_dir="/rds/user/nmdt2/hpc-work/outputs/20230725_sft_0",
            num_train_epochs=3,
            optim="paged_adamw_8bit"
        ),
        callbacks=callbacks
    )

    trainer.train()
    
    trainer.model.save_pretrained("./outputs/sft")

if __name__ == "__main__":
    finetune()
