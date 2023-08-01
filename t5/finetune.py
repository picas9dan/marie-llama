import os

from datasets import Dataset
import transformers
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from t5.arguments_schema import DatasetArguments, ModelArguments
from t5.dataset_utils import load_dataset


def train():
    hfparser = transformers.HfArgumentParser((ModelArguments, DatasetArguments, Seq2SeqTrainingArguments))
    model_args, data_args, train_args = hfparser.parse_args_into_dataclasses()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)

    dataset = load_dataset(data_args.data_path)

    def _tokenize(examples):
        model_inputs = tokenizer(
            examples["source"], 
            max_length=data_args.source_max_len,
            truncation=True
        )
        labels = tokenizer(
            examples["target"],
            max_length=data_args.target_max_len,
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_ds = dataset.map(_tokenize, batched=True, remove_columns=["source", "target"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model_output_dir = os.path.join(train_args.output_dir, "model")
    trainer.model.save_pretrained(model_output_dir)
    trainer.tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    train()
