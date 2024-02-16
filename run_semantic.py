"""
This example uses Approximate Nearest Neighbor Search (ANN) with Hnswlib  (https://github.com/nmslib/hnswlib/).

Searching a large corpus with Millions of embeddings can be time-consuming. To speed this up,
ANN can index the existent vectors. For a new query vector, this index can be used to find the nearest neighbors.

This nearest neighbor search is not perfect, i.e., it might not perfectly find all top-k nearest neighbors.
"""
from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import sys
import hnswlib
from pympler import asizeof

class SemanticSearch:
    def __init__(self):
        print('Loading model...')
        model_name = "sentence-transformers/msmarco-distilbert-base-tas-b"
        self.model = SentenceTransformer(model_name)
        print('Model Successfully Loaded!')

        self.embedding_cache_path = "document_embeddings.pkl"

        self.embedding_size = 768  # Size of embeddings
        self.top_k_hits = 10  # Output k hits


        print("Load pre-computed embeddings from disc")
        with open(self.embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            print(asizeof.asized(cache_data))
            self.corpus_sentences = cache_data["sentences"]
            self.corpus_embeddings = cache_data["embeddings"]
        # Defining our hnswlib index
        index_path = "./hnswlib.index"
        # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length, then is Inner Product equal to cosine similarity
        self.index = hnswlib.Index(space="cosine", dim=self.embedding_size)

        if os.path.exists(index_path):
            print("Loading index...")
            self.index.load_index(index_path)
            print(asizeof.asized(self.index))
        else:
            ### Create the HNSWLIB index
            print("Start creating HNSWLIB index")
            self.index.init_index(max_elements=len(self.corpus_embeddings), 
                             ef_construction=400, M=64)

            # Then we train the index to find a suitable clustering
            self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))

            print("Saving index to:", index_path)
            self.index.save_index(index_path)
            # Controlling the recall by setting ef:
            self.index.set_ef(50)  # ef should always be > top_k_hits


    def run_search(self, inp_question):
        #Search in the index
        print("Corpus loaded with {} sentences / embeddings"
              .format(len(self.corpus_sentences)))
        start_time = time.time()
        question_embedding = self.model.encode(inp_question)

        # We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(question_embedding, 
                                                     k=self.top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [{"corpus_id": id, "score": 1 - score} 
                for id, score in zip(corpus_ids[0], distances[0])]

        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        end_time = time.time()

        print("Input question:", inp_question)
        print("Results (after {:.3f} seconds):".format(end_time - start_time))
        for hit in hits[0:self.top_k_hits]:
            print("\t{:.3f}\t{}".format(hit["score"], self.corpus_sentences[hit["corpus_id"]]))

        # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
        # Here, we compute the recall of ANN compared to the exact results
        correct_hits = util.semantic_search(question_embedding, 
                                            self.corpus_embeddings, 
                                            top_k=self.top_k_hits)[0]
        correct_hits_ids = set([hit["corpus_id"] for hit in correct_hits])

        ann_corpus_ids = set([hit["corpus_id"] for hit in hits])
        if len(ann_corpus_ids) != len(correct_hits_ids):
            print("Approximate Nearest Neighbor returned a different number of results than expected")

        recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
        print("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(
            self.top_k_hits, recall * 100))

        if recall < 1:
            print("Missing results:")
            for hit in correct_hits[0:self.top_k_hits]:
                if hit["corpus_id"] not in ann_corpus_ids:
                    print("\t{:.3f}\t{}".format(
                        hit["score"], self.corpus_sentences[hit["corpus_id"]]))
        print("\n\n========\n")

    def run_search_modified(self, inp_question):
        #Search in the index
        print("Corpus loaded with {} sentences / embeddings"
              .format(len(self.corpus_sentences)))
        start_time = time.time()
        question_embedding = self.model.encode(inp_question)

        # We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(question_embedding,
                                                     k=self.top_k_hits)

        # We extract corpus ids and scores for the first query
        hits = [{"corpus_id": id, "score": 1 - score}
                for id, score in zip(corpus_ids[0], distances[0])]

        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        end_time = time.time()

        output = f"Input question:{inp_question}<br>Results (after {(end_time - start_time):.3f} seconds):<br>"
        for hit in hits[0:self.top_k_hits]:
            output += "{} - {:.3f}<br>".format(self.corpus_sentences[hit["corpus_id"]], hit["score"])
        return output
