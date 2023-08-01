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


PREFIX = "translate to SPARQL: "

def train():
    hfparser = transformers.HfArgumentParser((ModelArguments, DatasetArguments, Seq2SeqTrainingArguments))
    model_args, data_args, train_args = hfparser.parse_args_into_dataclasses()

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)

    dataset = Dataset.from_json(data_args.data_path)

    def preprocess_function(examples):
        sources = [PREFIX + example for example in examples["question"]]
        targets = examples["query"]
        return tokenizer(sources, text_target=targets, max_length=512, truncation=True)

    tokenized_ds = dataset.map(preprocess_function, batched=True, remove_columns=["question", "query"])

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
