import copy
from dataclasses import dataclass
import os
from typing import Dict, Sequence

from datasets import load_dataset
import transformers
import torch
from torch.nn.utils.rnn import pad_sequence

from arguments_schema import DataArgs, TrainArgs
from prompt_templates import ALPACA_TEMPLATE, MARIE_NO_CONTEXT_TEMPLATE, MARIE_WITH_CONTEXT_TEMPLATE


IGNORE_INDEX = -100


def preprocess_dataset(dataset, dataset_format: str):
    """Formats dataset to have only ("input", "output") columns."""
    if dataset_format == "alpaca":
        dataset = dataset.map(
            lambda example: {"input": ALPACA_TEMPLATE.format(**example)}, 
            remove_columns=["instruction"]
        )
    elif dataset_format == "self-instruct":
        dataset = dataset.rename_columns({"prompt": "input", "completion": "output"})
    elif dataset_format == "marie_no_context":
        dataset = dataset.map(
            lambda example: {"input": MARIE_NO_CONTEXT_TEMPLATE.format(**example), "output": example["query"]}
        )
    elif dataset_format == "marie_with_context":
        dataset = dataset.map(
            lambda example: {"input": MARIE_WITH_CONTEXT_TEMPLATE.format(**example), "output": example["query"]}
        )
    else:
        raise ValueError(f"Invalid dataset_format argument {dataset_format}.")
    
    dataset["train"] = dataset["train"].remove_columns([x for x in dataset["train"].column_names if x not in ("input", "output")])

    return dataset


def split_dataset(dataset, train_args: TrainArgs):
    def _get_example_length(example):
        return dict(length=len(example["input"]) + len(example["output"]))
    
    if train_args.do_eval:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            dataset = dataset["train"].train_test_split(
                test_size=0.1, shuffle=True
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

    return train_dataset, eval_dataset



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
    dataset = load_dataset(data_args.dataset, use_auth_token=os.getenv("HF_ACCESS_TOKEN"))
    preprocess_dataset(dataset, data_args.dataset_format)

    train_dataset, eval_dataset = split_dataset(dataset, train_args)

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