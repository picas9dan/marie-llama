import argparse
import json
from typing import List

from nltk.translate.bleu_score import corpus_bleu


def get_bleu_score(data: List[dict]):
    references = [[datum["gt"].split()] for datum in data]
    hypotheses = [datum["prediction"].split() for datum in data]

    return corpus_bleu(references, hypotheses)


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    bleu_score = get_bleu_score(data)
    print("BLEU score: ", bleu_score)

if __name__ == "__main__":
    eval()