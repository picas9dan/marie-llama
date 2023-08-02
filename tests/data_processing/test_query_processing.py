import pytest

from marie.data_processing.query_processing import (
    add_prefixes,
    decode_query,
    encode_query,
    remove_prefixes,
)


class TestQueryUtils:
    def test_encodeQuery(self):
        query = "SELECT *\nWHERE { ?s ?p ?o }\n"
        expected = "SELECT *\nWHERE  ob  var_s var_p var_o  cb \n"
        assert encode_query(query) == expected

    def test_decodeQuery(self):
        query = "SELECT *\nWHERE  ob  var_s var_p var_o  cb \n"
        expected = "SELECT *\nWHERE { ?s ?p ?o }\n"
        assert decode_query(query) == expected

    @pytest.mark.parametrize(
        "query, expected",
        [
            (
                (
                    "PREFIX os: <http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                    "SELECT *\nWHERE {?s ?p ?o}\n"
                ),
                "SELECT *\nWHERE {?s ?p ?o}\n",
            ),
            (
                (
                    "PREFIX os: \n<http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                    "PREFIX \nrdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                    "SELECT *\nWHERE {?s ?p ?o}\n"
                ),
                "SELECT *\nWHERE {?s ?p ?o}\n",
            ),
            (
                "SELECT *\nWHERE {?s ?p ?o}\n",
                "SELECT *\nWHERE {?s ?p ?o}\n",
            ),
        ],
    )
    def test_removePrefixes(self, query, expected):
        assert remove_prefixes(query) == expected

    def test_addPrefixes(self):
        query = "SELECT *\nWHERE {?s ?p ?o}\n"
        expected = (
            "PREFIX os: <http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "SELECT *\nWHERE {?s ?p ?o}\n"
        )
        assert add_prefixes(query) == expected

    def test_preprocessQuery(self):
        pass

    def test_postprocessQuery(self):
        pass
