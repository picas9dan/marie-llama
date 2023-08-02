from src.data_processing.qn_processing import preprocess_qn


class TestDatasetUtils:
    def test_preprocessQn(self):
        qn = "What is the molecular weight of ethanol?"
        expected = "translate to SPARQL: What is the molecular weight of ethanol?"
        actual = preprocess_qn(qn)
        assert actual == expected

    def test_preprocessQuery(self):
        pass

    def test_preprocessExamples(self):
        pass

    def test_loadDataset(self):
        pass