import concurrent.futures
import pickle
from typing import Callable

import numpy as np
from Stemmer import Stemmer

from KeywordSearch.loader import stopwords_set, token_dir, stemmer
from KeywordSearch.utils import cast2intarr

def build_inverted_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: tuple[tuple]):
    # todo: variable dtype
    with open(fname, 'r') as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    tokens_arr, tokens_delta = cast2intarr(tokens)
    for token in set(tokens):
        token_occurences = np.where(tokens_arr == (token - tokens_delta))[0]
        index[token][book_id] = cast2intarr(token_occurences)

def build_bow_index_alt(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[dict]):
    # todo: variable dtype
    with open(fname, 'r') as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    vocab_list = sorted(set(tokens))
    vocab, vocab_delta = cast2intarr(vocab_list)
    counts, count_delta = cast2intarr([tokens.count(token) for token in vocab_list])
    index[book_id] = (vocab_delta, vocab, count_delta, counts)

def build_bow_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[tuple]):
    # todo: variable dtype
    with open(fname, 'r') as f:
        tokens = np.array([token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set])
    vocab_list = sorted(set(tokens))
    vocab, vocab_delta = cast2intarr(vocab_list)
    counts, count_delta = cast2intarr([np.sum(vocab == token) for token in vocab_list])
    index[book_id] = (vocab_delta, vocab, count_delta, counts)

def build_full_index(offset: int=0, k: int=-1, index_func: Callable=build_bow_index) -> list:
    with open("valid_books.pkl", "rb") as f:
        _, _, valid_books = pickle.load(f)
    with open("all_tokens.pkl", "rb") as f:
        _, _, all_tokens = pickle.load(f)

    if isinstance(index_func, str):
        index_func = {}
    all_tokens = tuple([''] + list(all_tokens)) # add a dummy token
    token_index_dict = dict((token, i) for i, token in enumerate(all_tokens))
    index = list(dict() for _ in all_tokens)
    book_path_template = token_dir + "PG%d_tokens.txt"
    
    list_length = len(valid_books)
    if offset >= list_length:
        return
    valid_books = sorted(valid_books)[offset:min(list_length, offset + k)]

    complete_counter = 0
    failed_jobs = []
    with concurrent.futures.ThreadPoolExecutor() as pool:
        jobs = {
            pool.submit(
                index_func, book_path_template % book_id, book_id, stemmer, token_index_dict, index)
                : book_id for book_id in valid_books
            }
        
        for job in concurrent.futures.as_completed(jobs):
            book_id = jobs[job]
            try:
                job.result()
            except Exception as e:
                raise e
                failed_jobs.append(book_id)
            complete_counter += 1
            print(f"Finished building index for {complete_counter} books...", end="\r")
    
    print(f"\n{len(failed_jobs)}/{len(jobs)} failures while building index")
    
    with open("index.pkl", "wb") as f:
        pickle.dump(index, f)
    
    return tuple(index), valid_books