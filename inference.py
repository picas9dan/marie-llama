import transformers
from transformers import GenerationConfig
import torch

from arguments_schema import DataArgs, GenArgs, InferArgs, ModelArgs
from dataset_utils import CausalLmCollator, CausalLmDataset
from model_utils import get_model_and_tokenizer
from torch.utils.data import DataLoader


def infer():
    hfparser = transformers.HfArgumentParser((ModelArgs, DataArgs, InferArgs))
    model_args, data_args,infer_args = hfparser.parse_args_into_dataclasses()

    model, tokenizer = get_model_and_tokenizer(model_args, is_train=False)
    model.eval()

    dataset = CausalLmDataset(
        data_args=data_args, 
        tokenizer=tokenizer, 
        is_train=False,
        is_supervised=False
    )
    data_collator = CausalLmCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=infer_args.batch_size, collate_fn=data_collator)

    predictions = []
    print("Generating predictions...")
    with torch.no_grad():
        for batch in dataloader:
            outputs = model.generate(
                **{k: v.to("cuda") for k, v in batch.items()},
                max_new_tokens=256,
            )
            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(output_texts)

    with open(infer_args.output_file, "w") as f:
        f.write("\n\n".join(predictions) + "\n")


if __name__ == "__main__":
    infer()