from SPARQLWrapper import SPARQLWrapper, JSON
from nl2sparql import Nl2SparqlModel


class Marie:
    def __init__(
        self,
        model_path: str = "google/flan-t5-base",
        kg_endpoint: str = "http://178.128.105.213:3838/blazegraph/namespace/ontospecies/sparql",
    ):
        self.nl2sparql_model = Nl2SparqlModel(model_path)

        sparql = SPARQLWrapper(kg_endpoint)
        sparql.setReturnFormat(JSON)
        self.sparql = sparql

    def execute_kg(self, query: str):
        self.sparql.setQuery(query)
        return self.sparql.queryAndConvert()["results"]["bindings"]

    def get_answer(self, question: str):
        query = self.nl2sparql_model(question)
        return self.execute_kg(query)
