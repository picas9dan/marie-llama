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
    if template_name in ("marie_no_context", "marie_with_context"):
        return "query"
    raise ValueError(f"Invalid template_name: {template_name}. Must be either `alpaca`, `marie_no_context`, or `marie_with_context`.")


class CausalLmDataset(Dataset):
    def __init__(self, data_args: DataArgs, tokenizer: transformers.PreTrainedTokenizer, is_train: bool, is_supervised: bool):
        """
        Args:
            data_path: Path to the dataset file.
            prompt_template: Prompt template to be populated by the dataset. Must be `alpaca`, `self-instruct`,
                `marie_no_context`, `marie_with_context`.
            tokenizer: Tokenizer to apply to the dataset.
        """
        super(CausalLmDataset, self).__init__()
        data_path = data_args.train_data_path if is_train else data_args.eval_data_path
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sources = [tokenizer.bos_token + PROMPT_TEMPLATES[data_args.prompt_template].format(**example) for example in data]
        if is_supervised:
            targets = [example[_get_target_col_name(data_args.prompt_template)] + tokenizer.eos_token for example in data]
        else:
            targets = ["" for _ in data]

        tokenized_sources = tokenizer(
            sources, 
            max_length=data_args.source_max_len, 
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = tokenizer(
            targets,
            max_length=data_args.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Construct input_ids and labels
        input_ids = []
        labels = []
        for source_input_ids, target_input_ids in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids.append(torch.tensor(source_input_ids + target_input_ids))
            labels.append(torch.tensor([IGNORE_INDEX for _ in range(len(source_input_ids))] + copy.deepcopy(target_input_ids)))

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

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to("cuda")
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).to("cuda")

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to("cuda")
        )


def get_data_module(data_args: DataArgs, tokenizer: transformers.PreTrainedTokenizer):
    train_dataset = CausalLmDataset(
        data_args=data_args, 
        tokenizer=tokenizer, 
        is_train=True, 
        is_supervised=True
    )
    eval_dataset = CausalLmDataset(
        data_args=data_args, 
        tokenizer=tokenizer, 
        is_train=False,
        is_supervised=True
    )
    data_collator = CausalLmCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
