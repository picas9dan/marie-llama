import json

import transformers
from tqdm.auto import tqdm

from marie.translation import TranslationModel

from marie.arguments_schema import DatasetArguments, InferenceArguments, ModelArguments


def rename_dict_keys(d: dict, mappings: dict):
    return {mappings[k] if k in mappings else k: v for k, v in d.items()}


def infer():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DatasetArguments, InferenceArguments)
    )
    model_args, data_args, infer_args = hfparser.parse_args_into_dataclasses()

    trans_model = TranslationModel(
        model_args.model_path, device="cuda", max_new_tokens=infer_args.max_new_tokens
    )

    with open(data_args.data_path, "r") as f:
        data = json.load(f)

    preds = []
    for datum in tqdm(data):
        pred = trans_model(datum["question"], postprocess=infer_args.postprocess)
        preds.append(pred)

    data_out = [
        {
            **rename_dict_keys(datum, dict(query="gt")),
            "prediction": pred,
        }
        for datum, pred in zip(data, preds)
    ]
    with open(infer_args.out_file, "w") as f:
        json.dump(data_out, f, indent=4)


if __name__ == "__main__":
    infer()
