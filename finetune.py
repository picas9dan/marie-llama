# Adapted from https://github.com/artidoro/qlora/blob/main/qlora.py

import copy
from dataclasses import dataclass
import json
import logging
import os
from typing import Dict, Sequence

from datasets import load_dataset
from peft import (
    get_peft_model, 
    LoraConfig, 
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Seq2SeqTrainer,
)
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch
from torch.nn.utils.rnn import pad_sequence

from arguments_schema import DataArgs, ModelArgs, TrainArgs


logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
MARIE_PROMPT = (
    "Below is a natural language question posed to The World Avatar's question answering system. "
    "Convert the question into a valid SPARQL query that can be made to The World Avatar's knowledge graphs.\n\n"
    "### Question:\n{question}\n\n### SPARQL query:"
)


def get_last_checkpoint(output_dir: str):
    """Retrieves the last checkpoint directory."""
    if not os.path.isdir(output_dir):
        return None # first training
    
    if os.path.exists(os.path.join(output_dir, "completed")): 
        print('Detected that training was already completed!')
        return None
    
    max_step = 0
    for filename in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, filename)) and filename.startswith("checkpoint"):
            max_step = max(max_step, int(filename[len("checkpoint-"):]))
    if max_step == 0:
        return None # training started, but no checkpoint
    
    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{max_step}')
    print(f"Found a previous checkpoint at: {checkpoint_dir}")

    return checkpoint_dir # checkpoint found!
    
    
def print_trainable_parameters(bits: int, model: transformers.PreTrainedModel):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if bits == 4: 
        trainable_params /= 2

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def get_model(model_args: ModelArgs, train_args: TrainArgs, checkpoint_dir: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=train_args.bits == 4,
        load_in_8bit=train_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_args.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        print("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, "adapter_model"), is_trainable=True)
    else:
        print("Adding LoRA modules.")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_args.lora_r,
            lora_alpha=train_args.lora_alpha,
            lora_dropout=train_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, config)

    model.config.use_cache = False
    print_trainable_parameters(model)
    print("Loaded model.")

    return model


def prepare_llama2(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    # Issue 1: vocab_size mismatch
    # https://github.com/huggingface/transformers/issues/24899

    # Issue 2: incorrect pad_token
    # For the tokenizer, the special tokens (bos, eos, unk, pad) are set to be (<s>, </s>, <unk>, <unk>).
    # tokenizer.decode([32000]) returns <pad>
    tokenizer.add_special_tokens(dict(pad_token="<pad>"))

    return tokenizer


@dataclass
class DataCollatorForCausalLM():
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int

    def __call__(self, instances: Sequence[dict]) -> Dict[str, torch.Tensor]:
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        
        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        
        input_ids, labels = [], []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources["input_ids"],
            tokenized_targets["input_ids"]
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            labels.append(
                torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            ) # mask out source tokens

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        data_dict = dict(
            input_ids=input_ids,
            attention_mask= input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels
        )

        return data_dict


def get_data_module(data_args: DataArgs, train_args: TrainArgs, tokenizer: transformers.PreTrainedTokenizer):
    dataset = load_dataset(data_args.dataset)

    # Format dataset to have only "input" and "output" columns
    if data_args.dataset_format == "alpaca":
        dataset = dataset.map(
            lambda example: {"input": ALPACA_PROMPT.format(**example)}, 
            remove_columns=["instruction"]
        )
    elif data_args.dataset_format == "self-instruct":
        dataset = dataset.rename_columns({"prompt": "input", "completion": "output"})
    else:
        raise ValueError(f"Invalid dataset_format argument {data_args.dataset_format}. ")

    def _get_example_length(example):
        return dict(length=len(example["input"]) + len(example["output"]))
    
    if train_args.do_eval:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            dataset = dataset["train"].train_test_split(
                test_size=data_args.eval_dataset_size, shuffle=True
            )
            eval_dataset = dataset["test"]
        if train_args.group_by_length:
            eval_dataset = eval_dataset.map(_get_example_length)
    else:
        eval_dataset = None
    
    if train_args.do_train:
        train_dataset = dataset["train"]
        if train_args.group_by_length:
            train_dataset = train_dataset.map(_get_example_length)
    else:
        train_dataset = None

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=data_args.source_max_len,
        target_max_len=data_args.target_max_len,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, **kwargs):
        fname = os.path.join(args.output_dir, "completed")
        with open(fname, "a"):
            os.utime(fname)

        self.save_model(args, state, kwargs)


def train():
    hfparser = transformers.HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    model_args, data_args, train_args, gen_args = hfparser.parse_args_into_dataclasses()
    
    checkpoint_dir = get_last_checkpoint(train_args.output_dir)

    model = get_model(model_args, train_args, checkpoint_dir)
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.base_model,
        padding_side="right",
        use_fast=False,
        use_auth_token=True
    )
    prepare_llama2(model, tokenizer)

    data_module = get_data_module(data_args, train_args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        **data_module,
    )
    trainer.add_callback(SavePeftModelCallback)

    all_metrics = dict(run_name=train_args.run_name)

    if train_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_metrics
        all_metrics.update(metrics)

    if train_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if train_args.do_train or train_args.do_eval:
        with open(os.path.join(train_args.output_dir, "metrics.json"), "w") as f:
            f.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()