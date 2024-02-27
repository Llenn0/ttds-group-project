"""
Scanning through a vast collection containing millions of embeddings can be a slow process. 
To expedite this, Approximate Nearest Neighbor (ANN) algorithms can organize the existing vectors into an index. 
When a new query vector is introduced, this index facilitates the identification of its closest counterparts.

However, this nearest neighbor identification process is not flawless, 
meaning it may not always accurately pinpoint all of the top-k closest neighbors.
"""
from sentence_transformers import SentenceTransformer, util
import pickle
import time
from pympler import asizeof


class SemanticSearch:
    def __init__(self):
        """
        constructor for SemanticSearch
        loads the model embeddings from pickle file
        """
        print('Loading model...')
        #loads in the model
        model_name = "sentence-transformers/msmarco-distilbert-base-tas-b"
        self.model = SentenceTransformer(model_name)
        print('Model Successfully Loaded!')

        self.embedding_cache_path = "document_embeddings.pkl"

        print("Load pre-computed embeddings from disc")
        #fetches the embeddings for all the english books and pickle load
        with open(self.embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            self.corpus_sentences = cache_data["sentences"]
            self.corpus_embeddings = cache_data["embeddings"]

        print("Corpus loaded with {} sentences / embeddings".format(
            len(self.corpus_sentences)))


    def runSearch(self, inp_question):
        """
        Takes in a string from the user and carries out embedding of the query,
        and calculation of similarity between document embeddings and query embedding.

        parameters: (str) inp_question
        return: [(int, str)] score
        """
        start_time = time.time()
        question_embedding = self.model.encode(inp_question)
        #get the util dot score from the question embedding and corpus embedding
        scores = util.dot_score(self.corpus_embeddings, question_embedding).cpu().tolist()
        end_time = time.time()
        print("Results (after {:.3f} seconds):".format(end_time - start_time))
        #produce the results in list format
        doc_score_pairs = list(zip(self.corpus_sentences, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        for doc, score in doc_score_pairs[:10]:
            print(score, doc)

        return doc_score_pairs
