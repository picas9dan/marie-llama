import argparse
import json

from nltk.translate.bleu_score import corpus_bleu


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    references = [[datum["gt"].split()] for datum in data]
    hypotheses = [datum["prediction"].split() for datum in data]

    bleu_score = corpus_bleu(references, hypotheses)
    print("BLEU score: ", bleu_score)

if __name__ == "__main__":
    eval()