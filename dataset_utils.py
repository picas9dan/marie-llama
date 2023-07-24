import copy
from dataclasses import dataclass
import json
from typing import Dict, Sequence

import transformers
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from arguments_schema import DataArgs
from prompt_templates import PROMPT_TEMPLATES


IGNORE_INDEX = -100


def _get_target_col_name(template_name: str):
    if template_name == "alpaca":
        return "output"
    if template_name in ("marie_no_context", "marie_with_context", "marie_no_context_v2", "simple_delimiter"):
        return "query"
    raise ValueError(f"Invalid template_name: {template_name}. Must be either `alpaca`, `marie_no_context`, or `marie_with_context`.")


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer):
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class SupervisedDataset(Dataset):
    def __init__(self, data_args: DataArgs, tokenizer: transformers.PreTrainedTokenizer, is_train: bool, is_supervised: bool):
        """
        Args:
            data_path: Path to the dataset file.
            prompt_template: Prompt template to be populated by the dataset. Must be `alpaca`, `self-instruct`,
                `marie_no_context`, `marie_with_context`.
            tokenizer: Tokenizer to apply to the dataset.
        """
        super(SupervisedDataset, self).__init__()
        data_path = data_args.train_data_path if is_train else data_args.eval_data_path
        with open(data_path, "r") as f:
            data = json.load(f)

        sources = [PROMPT_TEMPLATES[data_args.prompt_template].format(**example) for example in data]
        targets = [example[_get_target_col_name(data_args.prompt_template)] + tokenizer.eos_token for example in data]

        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX

        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

@dataclass
class CausalLmCollator():
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels = ([instance[x] for instance in instances] for x in ("input_ids", "labels"))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )


def get_data_module(data_args: DataArgs, tokenizer: transformers.PreTrainedTokenizer, do_eval: bool):
    train_dataset = SupervisedDataset(
        data_args=data_args, 
        tokenizer=tokenizer, 
        is_train=True, 
        is_supervised=True
    )
    eval_dataset = SupervisedDataset(
        data_args=data_args, 
        tokenizer=tokenizer, 
        is_train=False,
        is_supervised=True
    ) if (do_eval and data_args.eval_data_path) else None
    data_collator = CausalLmCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
