
from typing import List
from sacrebleu.metrics import BLEU
from sklearn.base import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def normalise_space(text: str):
    return " ".join(text.strip().split())


def get_bleu_metrics(refs: List[List[str]], sys: List[str]):
    bleu = BLEU()
    b = bleu.corpus_score(sys, refs)
    return dict(score=b.score, precisions=b.precisions, bp=b.bp, ratio=b.ratio, hyp_len=b.sys_len, ref_len=b.ref_len)


def get_translation_metrics(data: List[dict]):
    queries = [normalise_space(datum["sparql_query"]) for datum in data]
    predictions = [normalise_space(datum["prediction"]) for datum in data]

    return dict(
        bleu=get_bleu_metrics([queries], predictions),
        accuracy=accuracy_score(queries, predictions)
    )


def get_retrieval_performance_metrics(gt_list: List[list], predictions_list: List[list]):
    mlb = MultiLabelBinarizer()
    multilabel_encodings = mlb.fit_transform(gt_list + predictions_list)
    gt_encodings = multilabel_encodings[:len(gt_list)]
    pred_encodings = multilabel_encodings[len(gt_list):]

    accuracy = accuracy_score(gt_encodings, pred_encodings)

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(gt_encodings, pred_encodings, average="micro")
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(gt_encodings, pred_encodings, average="macro")

    return dict(
        accuracy=accuracy,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1
    )