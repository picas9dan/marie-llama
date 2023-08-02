from src.utils.str_utils import advance_idx_thru_space, advance_idx_to_kw


def encode_query(query: str):
    return query


def decode_query(query: str):
    return query


def remove_prefixes(query: str):
    idx = advance_idx_to_kw(query, "PREFIX")

    while query.startswith("PREFIX", idx):
        "PREFIX prefix: <iri>"
        idx += len("PREFIX")
        idx = advance_idx_to_kw(query, ">", idx)
        idx += len(">")
        idx = advance_idx_thru_space(query, idx)

    return query[idx:]


def preprocess_query(query: str):
    query = remove_prefixes(query)
    return encode_query(query)




