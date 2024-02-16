"""
This example uses Approximate Nearest Neighbor Search (ANN) with Hnswlib  (https://github.com/nmslib/hnswlib/).

Searching a large corpus with Millions of embeddings can be time-consuming. To speed this up,
ANN can index the existent vectors. For a new query vector, this index can be used to find the nearest neighbors.

This nearest neighbor search is not perfect, i.e., it might not perfectly find all top-k nearest neighbors.
"""
from sentence_transformers import SentenceTransformer, util
import pickle
import time
from pympler import asizeof


class SemanticSearch:
    def __init__(self):
        print('Loading model...')
        model_name = "sentence-transformers/msmarco-distilbert-base-tas-b"
        self.model = SentenceTransformer(model_name)
        print('Model Successfully Loaded!')

        self.embedding_cache_path = "document_embeddings.pkl"

        print("Load pre-computed embeddings from disc")
        with open(self.embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            self.corpus_sentences = cache_data["sentences"]
            self.corpus_embeddings = cache_data["embeddings"]

        print("Corpus loaded with {} sentences / embeddings".format(
            len(self.corpus_sentences)))


    def runSearch(self, inp_question):
        inp_question = input("Please enter a question: ")
        start_time = time.time()
        question_embedding = self.model.encode(inp_question)
        scores = util.dot_score(self.corpus_embeddings, question_embedding).cpu().tolist()
        end_time = time.time()
        print("Results (after {:.3f} seconds):".format(end_time - start_time))
        doc_score_pairs = list(zip(self.corpus_sentences, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        for doc, score in doc_score_pairs[:10]:
            print(score, doc)
