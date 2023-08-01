from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

QUERY_PREFIXES = (
    "PREFIX os: <http://www.theworldavatar.com/ontology/ontospecies/OntoSpecies.owl#>\n"
    "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
)


class Nl2SparqlModel:
    def __init__(
        self, 
        model_path: str="google/flan-t5-base", 
        device="cuda",
        max_new_token=512,
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.max_new_tokens=max_new_token


    def preprocess(self, question: str):
        question = "translate to SPARQL: " + question
        return question
    
    def postprocess(self, query: str):
        query = QUERY_PREFIXES + query
        return query
    
    def __call__(self, question: str):
        question = self.preprocess(question)
        
        input_ids = self.tokenizer(question, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens)
        query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return self.postprocess(query)