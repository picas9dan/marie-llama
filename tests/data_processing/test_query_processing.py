import pytest

from marie.data_processing.query_processing import remove_prefixes


class TestQueryUtils:
    def test_encodeQuery(self):
        pass

    def test_decodeQuery(self):
        pass

    @pytest.mark.parametrize(
        "query, expected",
        [
            (
                (
                    "PREFIX os: <http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                    "SELECT *\n"
                    "WHERE {?s ?p ?o}\n"
                ),
                "SELECT *\nWHERE {?s ?p ?o}\n",
            ),
            (
                (
                    "PREFIX os: \n<http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
                    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                    "PREFIX \nrdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
                    "SELECT *\n"
                    "WHERE {?s ?p ?o}\n"
                ),
                "SELECT *\nWHERE {?s ?p ?o}\n",
            ),
        ],
    )
    def test_removePrefixes(self, query, expected):
        assert remove_prefixes(query) == expected

    def test_preprocessQuery(self):
        pass
