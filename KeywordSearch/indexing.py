from typing import Iterable

import concurrent.futures
import pickle
import os
import traceback
import gc
import re
import glob
from collections import defaultdict

from tqdm.notebook import tqdm
import numpy as np
from Stemmer import Stemmer

from KeywordSearch.loader import stopwords_set, token_dir, stemmer, LOG_PATH
from KeywordSearch.utils import cast2intarr, save_in_batches

class ZeroDict(dict):
    def __missing__(self, _):
        return 0

def build_inverted_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[dict]):
    with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    tokens_arr = np.array(tokens)
    for token in set(tokens):
        token_occurences = np.where(tokens_arr == token)[0]
        index[token][book_id], _ = cast2intarr(token_occurences, delta_encode=False)
    del tokens_arr, token_occurences

def build_inverted_index_batch(job_infos: list[tuple[str, int]], stemmer: Stemmer, token_index_dict: dict, index: list[dict]):
    for book_id, fname in job_infos:
        with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
            tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                    if token not in stopwords_set]
        tokens_arr = np.array(tokens)
        for token in set(tokens):
            token_occurences = np.where(tokens_arr == token)[0]
            index[token][book_id], _ = cast2intarr(token_occurences, delta_encode=False)
    del tokens, tokens_arr, token_occurences
    gc.collect()

def build_bow_index_alt(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[dict]):
    with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    vocab_list = sorted(set(tokens))
    vocab, vocab_delta = cast2intarr(vocab_list)
    counts, count_delta = cast2intarr([tokens.count(token) for token in vocab_list])
    index[book_id] = (vocab_delta, vocab, count_delta, counts)

def build_bow_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: list[tuple]):
    with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
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
        index_func = build_inverted_index_batch
        index_size = len(all_tokens)

    complete_counter = 0
    failed_jobs = []

    if True:
        valid_books_batch = []
        num_books = len(valid_books)
        num_batches = num_books // 20
        end = 0
        i = -1
        for i in range(num_batches):
            start = i * 20
            end = min(start + 20, num_books)
            valid_books_batch.append(tuple((book_id, book_path_template % book_id) for book_id in valid_books[start:end]))
        
        if end != num_books:
            valid_books_batch.append(tuple((book_id, book_path_template % book_id) for book_id in valid_books[end:]))

    with concurrent.futures.ThreadPoolExecutor() as pool:
        # jobs = {
        #     pool.submit(
        #         index_func, book_path_template % book_id, book_id, stemmer, token_index_dict, index.copy())
        #         : book_id for book_id in valid_books
        #     }
        jobs = {
            pool.submit(
                index_func, books_info, stemmer, token_index_dict, index)
                : [info[0] for info in books_info] for books_info in valid_books_batch
            }
        
        print(f"{len(jobs)} jobs in total")

        for job in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs)):
            book_id = jobs[job]
            try:
                job.result()
                # for token_entry, addition in zip(index, job.result()):
                #     if addition:
                #         token_entry.update(addition)
            except Exception as e:
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"Create index failure at book {book_id}:\n{''.join(traceback.format_exception(e))}\n")
                failed_jobs.append(book_id)
            complete_counter += 1
            if complete_counter % 5 is 0: gc.collect()
            #print(f"Finished building index for {complete_counter} books...", end="\r")
        
        # concurrent.futures.wait(jobs)
        #print(f"Finished building index for {complete_counter} books...", flush=True)
    print(f"{len(failed_jobs)}/{len(jobs)} failures while building index", flush=True)

    # done_jobs = set(jobs.values())
    done_jobs = []
    for job_batch in jobs.values():
        done_jobs.extend(job_batch)
    done_jobs = set(done_jobs)
    del jobs
    gc.collect()
    
    if not skip_pickle:
        if not os.path.exists("index"):
            os.mkdir("index")
        try:
            with open(f"done_jobs_{prefix}.pkl", "wb") as f:
                pickle.dump(done_jobs, f)
            save_in_batches(batch_size, index_type, index, prefix, index_size, unsafe_pickle)
        except Exception as e:
            print(e)
            print("Pickle Failure")
        gc.collect()
    return index, valid_books

def merge_parts(parts: list[str], segment_index: str | int=-1, save_dir: str="index"):
    with open(parts.pop(), "rb") as f:
        index: Iterable[dict] = pickle.load(f)
    for part in parts:
        with open(part, "rb") as f:
            tmp = pickle.load(f)
        for token, new_part in zip(index, tmp):
            token.update(new_part)
        gc.collect()
    with open(os.path.join(save_dir, f"{segment_index}_merged.pkl"), "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    del index
    gc.collect()

def merge_index(dir: str="index", naming_rule: str=r"part([0-9]+)_inverted_([0-9]+).pkl"):
    naming_regex = re.compile(naming_rule)
    index_segments = glob.glob(naming_rule.replace("([0-9]+)", '*'), root_dir=dir)
    lookup_table = defaultdict(lambda : list())
    for segment_name in index_segments:
        match = naming_regex.fullmatch(segment_name)
        if match:
            segment_index = match.group(2)
            lookup_table[segment_index].append(os.path.join(dir, segment_name))

    print(f"{len(lookup_table)} segments to merge")
    failed_jobs = []
    complete_counter = 0
    with concurrent.futures.ProcessPoolExecutor() as pool:
        jobs = {pool.submit(merge_parts, segments, segment_index, dir) : segment_index
                for segment_index, segments in lookup_table.items()}
        for job in concurrent.futures.as_completed(jobs):
            segment_index = jobs[job]
            try:
                job.result()
            except Exception as e:
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"Merge index failure at segment {segment_index}:\n{''.join(traceback.format_exception(e))}\n")
                failed_jobs.append(segment_index)
            complete_counter += 1
            print(f"Finished merging parts for {complete_counter} segments...", end="\r", flush=True)
    print("\nAll done")