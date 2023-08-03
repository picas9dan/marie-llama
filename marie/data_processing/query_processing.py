
from marie.utils import advance_idx_thru_space, advance_idx_to_kw


QUERY_ENCODINGS = {
    "{": " ob ",
    "}": " cb ",
    "?": "var_"
}
QUERY_DECODINGS = {v: k for k, v in QUERY_ENCODINGS.items()}


def encode_query(query: str):
    for k, v in QUERY_ENCODINGS.items():
        query = query.replace(k, v)

    return query


def decode_query(query: str):
    for k, v in QUERY_ENCODINGS.items():
        query = query.replace(v, k)

    return query


def remove_prefixes(query: str):
    idx = advance_idx_to_kw(query, "PREFIX")
    if idx == len(query):
        return query

    while query.startswith("PREFIX", idx):
        "PREFIX prefix: <iri>"
        idx += len("PREFIX")
        idx = advance_idx_to_kw(query, ">", idx)
        idx += len(">")
        idx = advance_idx_thru_space(query, idx)

    return query[idx:]


def preprocess_query(query: str):
    query = remove_prefixes(query)
    query = encode_query(query)
    return query


def postprocess_query(query: str):
    query = decode_query(query)
    return query
