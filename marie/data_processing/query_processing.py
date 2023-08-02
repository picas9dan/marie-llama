
from marie.utils import advance_idx_thru_space, advance_idx_to_kw


QUERY_ENCODINGS = {
    "{": "ob",
    "}": "cb"
}

QUERY_PREFIXES = (
    "PREFIX os: <http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
)


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


def add_prefixes(query: str):
    return QUERY_PREFIXES + query


def preprocess_query(query: str):
    query = remove_prefixes(query)
    query = encode_query(query)
    return query


def postprocess_query(query: str):
    query = decode_query(query)
    query = add_prefixes(query)
    return query
