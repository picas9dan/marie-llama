import json

from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

def infer():
    base_model = "google/flan-t5-base"
    model_path = "./outputs/model"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    translator = pipeline(
        "translation",
        model=model, 
        tokenizer=tokenizer,
        max_length=512,
        device_map={"": 0}
    )

    dataset = Dataset.from_json("./data/test_20230724.json")

    predictions = []
    for out in tqdm(translator(KeyDataset(dataset, "question"))):
        predictions.append(out)

    with open("./predictions_20230728.json", "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    infer()