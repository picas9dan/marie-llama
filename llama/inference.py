import json

from datasets import Dataset
import torch
import transformers
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

from llama.arguments_schema import DatasetArguments, InferenceArguments, ModelArguments
from model_utils import get_model, get_tokenizer
from prompt_templates import TEMPLATES


def format_prompt(template_name: str, example: dict):
    example["prompt"] = TEMPLATES[template_name]["prompt"].format(**example)
    return example


def infer():
    hfparser = transformers.HfArgumentParser((ModelArguments, DatasetArguments, InferenceArguments))
    model_args, data_args, infer_args = hfparser.parse_args_into_dataclasses()

    model = get_model(model_args)
    model.eval()

    tokenizer = get_tokenizer(model_args.base_model)

    dataset = Dataset.from_json(data_args.data_path)
    dataset = dataset.map(format_prompt)
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        eos_token_id=tokenizer.eos_token_id,
        max_length=512,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )

    predictions = []
    for out in tqdm(pipe(KeyDataset(dataset, "prompt")), batch_size=infer_args.batch_size):
        predictions.append(out)

    with open(infer_args.output_file, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    infer()