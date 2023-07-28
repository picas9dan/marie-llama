import os

from datasets import Dataset
from peft import LoraConfig
import transformers
from transformers import TrainerCallback, TrainingArguments
from trl import SFTTrainer

from llama.arguments_schema import DatasetArguments, ModelArguments
from model_utils import get_model, get_tokenizer

from prompt_templates import TEMPLATES


def example_formatter_gen(template_name: str):
    template = TEMPLATES[template_name]["prompt"] + TEMPLATES[template_name]["completion"]

    def construct_examples(examples: dict):
        output_texts = []

        for i in range(len(examples["question"])):
            example = {k: v[i] for k, v in examples.items()}        
            text = template.format(**example)
            output_texts.append(text)

        return output_texts

    return construct_examples


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print("Saving PEFT checkpoint...")
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)


def train():
    hfparser = transformers.HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    model_args, data_args, train_args = hfparser.parse_args_into_dataclasses()

    model = get_model(model_args)
    model.config.use_cache = False 
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    tokenizer = get_tokenizer(model_args.base_model)
    dataset = Dataset.from_json(data_args.data_path)
    formatting_func = example_formatter_gen(data_args.prompt_template)
    callbacks = [PeftSavingCallback]

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=train_args,
        callbacks=callbacks
    )
    trainer.train()


if __name__ == "__main__":
    train()
