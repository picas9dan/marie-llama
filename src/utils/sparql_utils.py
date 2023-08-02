from src.utils.str_utils import advance_idx_thru_space, advance_idx_to_kw


def preprocess_query(query: str):
    # removes prefix if any
    return encode(query)


def remove_prefixes(query: str):
    idx = advance_idx_to_kw(query, "PREFIX")

    while query.startswith("PREFIX", idx):
        "PREFIX prefix: <iri>"
        idx += len("PREFIX")
        idx = advance_idx_to_kw(query, ">", idx)
        idx = advance_idx_thru_space(query, idx)
    
    return query[idx:]


def encode(query: str):
    return query


def decode(query: str):
    return query