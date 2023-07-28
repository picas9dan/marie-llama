import argparse
import json
from typing import List

from nltk.translate.bleu_score import corpus_bleu
from SPARQLWrapper import SPARQLWrapper, JSON


SPARQL_ENDPOINT = "http://178.128.105.213:3838/blazegraph/namespace/ontospecies/sparql"
QUERY_PREFIX = (
    "PREFIX os: <http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "\n"
    "SELECT DISTINCT ?label "
)

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.setReturnFormat(JSON)


def get_bleu_score(data: List[dict]):
    references = [[datum["gt"].split()] for datum in data]
    hypotheses = [datum["prediction"].split() for datum in data]

    return corpus_bleu(references, hypotheses)


def execute_queries(queries: List[str]):
    results = []

    for query in queries:
        sparql.setQuery(query)

        try:
            result = sparql.queryAndConvert()
            result = results["results"]["bindings"]
            results.append(result)
        except Exception as e:
            results.append(e)
    
    return results


def populate_answers(data: dict):
    if "gt_answer" not in data[0]:
        gt_answers = execute_queries([QUERY_PREFIX + datum["gt"] for datum in data])
        for (datum, answer) in zip(data, gt_answers):
            datum["gt_answer"] = answer
    else:
        gt_answers = [datum["gt_answer"] for datum in data]
    
    predicted_answers = execute_queries([QUERY_PREFIX + datum["prediction"] for datum in data])
    for (datum, answer) in zip(data, predicted_answers):
        datum["predicted_answer"] = answer
    
    return gt_answers, predicted_answers


def get_acc(gt: list, preds: list):
    assert len(gt) == len(preds)
    return sum(x == y for x, y in zip(gt, preds)) / len(gt)


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    bleu_score = get_bleu_score(data)
    gt_answers, predicted_answers = populate_answers(data)

    results = dict(
        bleu_score=bleu_score,
        acc=get_acc(gt_answers, predicted_answers),
        data=data
    )
    with open(args.output_file, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    eval()