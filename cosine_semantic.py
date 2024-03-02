from sentence_transformers import SentenceTransformer, util
import pickle
import time

class SemanticSearch:
    def __init__(self):
        """
        Constructor for SemanticSearch.
        Loads the model embeddings from a pickle file.
        """
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

        print("Corpus loaded with {} sentences / embeddings".format(len(self.corpus_sentences)))

    def runSearch(self, inp_question):
        """
        Takes in a string from the user and carries out embedding of the query,
        and calculation of similarity between document embeddings and query embedding.

        Parameters:
            inp_question (str): The input question.

        Returns:
            List[Tuple[int, str]]: A list of scores.
        """
        start_time = time.time()
        question_embedding = self.model.encode(inp_question, convert_to_tensor=True).cpu()
        # Calculate cosine similarity scores
        scores = util.cos_sim(self.corpus_embeddings, question_embedding).cpu().tolist()

        end_time = time.time()
        print("Results (after {:.3f} seconds):".format(end_time - start_time))

        # Process and print the results
        doc_score_pairs = []
        for i in range(len(scores)):
            for j in range(len(scores[i])):
                doc_score_pairs.append((self.corpus_sentences[i], scores[i][j]))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        for doc, score in doc_score_pairs[:10]:
            print(score, doc)

        return doc_score_pairs

test = SemanticSearch()
test.runSearch("sciences fiction")
