from datasets import Dataset

PREFIX = "translate to SPARQL: "


def preprocess_qn(qn: str):
    qn = PREFIX + qn
    # convert units
    return qn


def preprocess_query(query: str):
    # condense graph pattern
    return query


def preprocess_examples(examples):
    sources = [preprocess_qn(qn) for qn in examples["question"]]
    targets = [preprocess_query(query) for query in examples["query"]]
    return dict(source=sources, target=targets)


def load_dataset(data_path: str):
    """"Loads and pre-processes a dataset with headers ("question", "query").
    Returns:
        A dataset with headers ("source", "target").
    """
    dataset = Dataset.from_json(data_path)
    dataset = dataset.map(preprocess_examples, batched=True, remove_columns=["question", "query"])
    return dataset