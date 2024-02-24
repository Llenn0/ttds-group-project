from collections import defaultdict
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import os
import pprint
import sys
from sentence_transformers import SentenceTransformer, util
import pickle

class SemanticSearch:

    def read_all_english_ids(self, filename):
        book_ids = []
        with open(filename, 'r') as file:
            for line in file.readlines():
                book_ids.append(line.replace('\n', '_text.txt'))
        return book_ids


    def check_if_books_exists(self, book_ids, directory):
        valid_book_file = []
        for filename in os.scandir(directory):
            id_check = filename.name
            if filename.is_file() and id_check in book_ids:
                valid_book_file.append(filename.name)
        return valid_book_file


    def save_to_embeddings(self, model, id_english_books):
        embedding_cache_path = "document_embeddings.pkl"
        book_ids = []
        books_content = []
        directory = './data/text/'
        for filename in os.scandir(directory):
            if filename.name in id_english_books:
                try:
                    with open(filename.path, 'r', encoding='utf-8') as file:
                        book_ids.append(filename.name.replace("_text.txt", ""))
                        books_content.append(file.read().replace('\n', ''))
                        #Load the model
                        model.encode(list(file.read().replace('\n', '')), 
                                     show_progress_bar=True, convert_to_numpy=True)
                except Exception as e:
                    print(f"Error reading {filename.name}: {e}")
        corpus_embeddings = model.encode(books_content, show_progress_bar=True, convert_to_numpy=True)
        print("Store file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({"sentences": book_ids, "embeddings": corpus_embeddings}, fOut)

if __name__ == '__main__':
    #initalise sherries preprocessing book
    searcher = SemanticSearch()
    #ebooks = searcher.read_all_books(None)
    ebook_ids = searcher.read_all_english_ids("./english_books.txt")
    valid_txt_books = searcher.check_if_books_exists(ebook_ids, './data/text')
    model = SentenceTransformer(
            'sentence-transformers/msmarco-distilbert-base-tas-b')
    searcher.save_to_embeddings(model, valid_txt_books[:30000])

