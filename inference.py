import json
import transformers
from transformers import GenerationConfig
import torch

from arguments_schema import DataArgs, GenArgs, ModelArgs
from model_utils import get_model_and_tokenizer
from prompt_templates import PROMPT_TEMPLATES


def infer():
    hfparser = transformers.HfArgumentParser((ModelArgs, DataArgs, GenArgs))
    model_args, data_args, gen_args = hfparser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer(model_args)
    model.eval()

    with open(data_args.eval_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = [tokenizer.bos_token + PROMPT_TEMPLATES[data_args.prompt_template].format(**example) for example in data]
    tokenized_sources = tokenizer(
        sources,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    tokenized_sources = {k: v.to("cuda") for k, v in tokenized_sources.items()}
    
    gen_config = GenerationConfig(**vars(gen_args))

    with torch.no_grad():
        outputs = model.generate(
            **tokenized_sources,
            generation_config=gen_config
        )

    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions = [text.split("### Query:", 1)[-1] for text in output_texts]

    with open("predictions.txt", "w") as f:
        f.writelines(predictions)


if __name__ == "__main__":
    infer()