from datasets import Dataset
import transformers
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)


PREFIX = "translate to structured query: "

def train():
    model_id = "google/flan-t5-base"
    data_path = "./data/train_20230721.json"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = Dataset.from_json(data_path)


    def preprocess_function(examples):
        sources = [PREFIX + example for example in examples["question"]]
        targets = examples["query"]
        return tokenizer(sources, text_target=targets, max_length=512, truncation=True)

    tokenized_ds = dataset.map(preprocess_function, batched=True, remove_columns=["question", "query"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./outputs",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        bf16=True,
        optim="paged_adamw_8bit"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.model.save_pretrained("./outputs/model")


if __name__ == "__main__":
    train()
