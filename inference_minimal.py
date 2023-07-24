import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset


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

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0

    data = Dataset.from_json("./data/test_20230724.json")
    inputs = [
        tokenizer(text + "\n\n###\n\n", return_tensors="pt") 
        for text in data["question"]
    ]
    with torch.no_grad():
        outputs = [
            model.generate(
                input_ids=x["input_ids"].to("cuda"), max_new_tokens=256
            ) for x in inputs
        ]

    with open("./predictions_20230724_minimal.txt", "r") as f:
        f.write("\n\n".join(outputs) + "\n")

if __name__ == "__main__":
    infer()