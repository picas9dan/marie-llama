import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import transformers
from transformers.pipelines.pt_utils import KeyDataset
from peft import PeftModel
from datasets import Dataset
from tqdm.auto import tqdm


def add_delimiter(example):
    example["question"]  = example["question"] + "\n\n###\n\n"
    return example


def infer():
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

    model = PeftModel.from_pretrained(model, "/rds/user/nmdt2/hpc-work/outputs/20230724_minimal/adapter_model")
    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN"),
    )
    tokenizer.pad_token_id = 0

    dataset = Dataset.from_json("./data/test_20230724.json")
    dataset = dataset.map(add_delimiter)
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )

    predictions = []
    for out in tqdm(pipe(KeyDataset(dataset, "question"))):
        predictions.append(out)

    with open("./predictions_20230724_minimal.txt", "r") as f:
        f.write("\n\n".join(predictions) + "\n")

if __name__ == "__main__":
    infer()