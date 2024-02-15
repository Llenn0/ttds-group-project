import concurrent.futures
import pickle
import os
import traceback
import gc

import numpy as np
from Stemmer import Stemmer

from KeywordSearch.loader import stopwords_set, token_dir, stemmer
from KeywordSearch.utils import cast2intarr, save_in_batches

class ZeroDict(dict):
    def __missing__(self, _):
        return 0

def build_inverted_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: tuple[tuple]):
    # todo: variable dtype
    with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    tokens_arr = np.array(tokens)
    for token in set(tokens):
        token_occurences = np.where(tokens_arr == token)[0]
        index[token][book_id], _ = cast2intarr(token_occurences, delta_encode=False)

def build_bow_index_alt(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[dict]):
    # todo: variable dtype
    with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    vocab_list = sorted(set(tokens))
    vocab, vocab_delta = cast2intarr(vocab_list)
    counts, count_delta = cast2intarr([tokens.count(token) for token in vocab_list])
    index[book_id] = (vocab_delta, vocab, count_delta, counts)

def build_bow_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[tuple]):
    # todo: variable dtype
    with open(fname, 'r') as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    vocab_list = sorted(set(tokens))
    token_arr = np.array(tokens)
    vocab, vocab_delta = cast2intarr(vocab_list)
    counts, count_delta = cast2intarr([np.sum(token_arr == token) for token in vocab_list])
    index[book_id] = (vocab_delta, vocab, count_delta, counts)

def build_full_index(offset: int=0, k: int=-1, batch_size: int=500, index_type: str="bow", prefix: str="", unsafe_pickle: bool=False, skip_pickle: bool=False) -> tuple[list, list]:
    assert index_type in ("bow", "inverted"), "index_type must be \"bow\" or \"inverted\""

    with open("valid_books.pkl", "rb") as f:
        _, _, valid_books = pickle.load(f)
    with open("all_tokens.pkl", "rb") as f:
        _, _, all_tokens = pickle.load(f)

    all_tokens = tuple(all_tokens)
    token_index_dict = ZeroDict((token, i) for i, token in enumerate(all_tokens))
    book_path_template = token_dir + "PG%d_tokens.txt"
    
    list_length = len(valid_books)
    if offset >= list_length:
        return
    valid_books = sorted(valid_books)[offset:min(list_length, offset + k)]

    # todo: switch between two modes
    if index_type == "bow":
        index = [None] * (np.max(valid_books) + 1)
        index_func = build_bow_index
        index_size = len(valid_books)
    else:
        index = [dict() for _ in all_tokens]
        index_func = build_inverted_index
        index_size = len(all_tokens)

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
                with open("log", 'a', encoding="UTF-8") as f:
                    f.write(f"Create index failure at book {book_id}:\n{''.join(traceback.format_exception(e))}\n")
                failed_jobs.append(book_id)
            complete_counter += 1
            if (complete_counter % 1000 is 0):
                gc.collect()
            print(f"Finished building index for {complete_counter} books...", end="\r")
        
        # concurrent.futures.wait(jobs)
        print(f"Finished building index for {complete_counter} books...", flush=True)
    print(f"{len(failed_jobs)}/{len(jobs)} failures while building index", flush=True)

    done_jobs = set(jobs.values())
    del jobs
    gc.collect()
    
    if not skip_pickle:
        try:
            with open(f"done_jobs_{prefix}.pkl", "wb") as f:
                pickle.dump(done_jobs, f)
            save_in_batches(batch_size, index_type, index, prefix, index_size, unsafe_pickle)
        except Exception as e:
            print(e)
            print("Pickle Failure")
        gc.collect()
    return index, valid_books