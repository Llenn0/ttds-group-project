from typing import Iterable

import concurrent.futures
import pickle
import os
import traceback
import gc
import re
import glob
from collections import defaultdict

import numpy as np
from Stemmer import Stemmer

from KeywordSearch.loader import token_dir, LOG_PATH, index_dir, tqdm
from KeywordSearch.utils import cast2intarr, save_in_batches, ZeroDict, save_json

def return_dict():
    return dict()

def build_inverted_index_batch(job_infos: list[int], token_index_dict: dict, 
                               index: list[dict] | None=None) -> list[dict] | defaultdict[int, dict]:
    if index is None:
        index = defaultdict(return_dict)
    
    if len(job_infos):
        tokenised_text_dir = token_dir + "PG%d_tokens.txt"
        for book_id in job_infos:
            with open(tokenised_text_dir % book_id, 'r', encoding="UTF-8", errors="ignore") as f:
                tokens = [token_index_dict[token] for token in f.read().splitlines()]
            tokens_arr = np.array(tokens)
            for token in set(tokens):
                token_occurences = np.where(tokens_arr == token)[0]
                index[token][book_id], _ = cast2intarr(token_occurences, delta_encode=False)
            del tokens_arr, token_occurences
        del tokens
        gc.collect()

    return index

def count_document_length_batch(job_infos: list[int]) -> list[dict] | defaultdict[int, dict]:
    counts = [None] * len(job_infos)

    if counts:
        tokenised_text_dir = token_dir + "PG%d_tokens.txt"
        for i, book_id in enumerate(job_infos):
            with open(tokenised_text_dir % book_id, 'r', encoding="UTF-8", errors="ignore") as f:
                num_tokens = len(f.read().splitlines())
            counts[i] = num_tokens
        gc.collect()

    return counts

def fetch_all_doc_length(batch_size: int=50, **kwargs) -> dict[int, int]:
    books_to_count = [int(fname.split('_')[0][2:]) for fname in glob.glob("PG*_tokens.txt", root_dir=token_dir)]
    books_to_count.sort()

    batches = []
    num_books = len(books_to_count)
    num_batches = num_books // batch_size
    if num_books % batch_size:
        num_batches += 1
    
    for i in range(num_batches):
        start = i * batch_size
        end = min(num_books, start + batch_size)
        batches.append(books_to_count[start:end])
    
    length = {book_id : 0 for book_id in books_to_count}

    with concurrent.futures.ProcessPoolExecutor(**kwargs) as pool:
        with tqdm(total=num_books, desc="Counting books") as pbar:
            jobs = {pool.submit(count_document_length_batch, job_infos) : job_infos 
                    for job_infos in batches}
            
            for job in concurrent.futures.as_completed(jobs):
                books_counted = jobs[job]
                try:
                    for book_id, book_length in zip(books_counted, job.result()):
                        length[book_id] = book_length
                except Exception as e:
                    with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                        f.write(f"Count document length failure at {books_counted}:\n{''.join(traceback.format_exception(e))}\n")
                pbar.update(len(books_counted))
    
    return length

def build_full_index(pool: concurrent.futures.ProcessPoolExecutor, offset: int=0, k: int=-1, batch_size: int=50, index_type: str="inverted", 
                     prefix: str="", skip_save: bool=False, use_json: bool=False) -> tuple[list, list]:
    assert index_type in ("bow", "inverted"), "index_type must be \"bow\" or \"inverted\""
    use_process = isinstance(pool, concurrent.futures.ProcessPoolExecutor)
    assert use_process or isinstance(pool, concurrent.futures.ThreadPoolExecutor), \
        "Argument `pool` must be a ProcessPoolExecutor or ThreadPoolExecutor"

    with open("processed_books.pkl", "rb") as f:
        processed_books = pickle.load(f)
    with open("all_tokens.pkl", "rb") as f:
        _, _, all_tokens = pickle.load(f)

    all_tokens = tuple(all_tokens)
    token_index_dict = ZeroDict((token, i) for i, token in enumerate(all_tokens))
    
    list_length = len(processed_books)
    if offset >= list_length:
        return
    books_to_index = sorted(processed_books)[offset:min(list_length, offset + k)]

    index = [dict() for _ in all_tokens]
    index_func = build_inverted_index_batch
    index_size = len(all_tokens)

    completed_jobs = 0
    failed_jobs = []
    done_jobs = set()

    # with concurrent.futures.ProcessPoolExecutor() as pool:
    jobs = dict()
    index_to_submit = None if use_process else index
    with tqdm(total=len(books_to_index), desc="Submitting jobs") as pbar:
        list_length = len(books_to_index)
        num_batches = list_length // batch_size
        if list_length % batch_size:
            num_batches += 1
        for i in range(num_batches):
            start = i * batch_size
            end = min(list_length, start + batch_size)
            slice = books_to_index[start:end]
            jobs[pool.submit(index_func, slice, token_index_dict, index_to_submit)] = slice
            pbar.update(len(slice))
    
    with tqdm(total=len(books_to_index), desc="Initialising...") as pbar:        
        pbar.set_description("Indexing")

        for job in concurrent.futures.as_completed(jobs):
            books = jobs[job]
            try:
                if use_process:
                    for add_key, addition in job.result().items():
                        if addition:
                            index[add_key].update(addition)
                    del addition
                else:
                    job.result()
                done_jobs.update(books)
            except Exception as e:
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"Create index failure at {books}:\n{''.join(traceback.format_exception(e))}\n")
                failed_jobs.extend(books)
            job_size = len(books)
            completed_jobs += job_size
            pbar.update(job_size)
            if completed_jobs % 5 is 0: gc.collect()

            #print(f"Finished building index for {complete_counter} books...", end="\r")
        
        # concurrent.futures.wait(jobs)
        #print(f"Finished building index for {complete_counter} books...", flush=True)
    print(f"{len(failed_jobs)}/{len(books_to_index)} failures while building index", flush=True)

    del jobs
    gc.collect()
    
    if not skip_save:
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        try:
            if use_json:
                save_json(sorted(done_jobs), f"done_jobs_{prefix}.json")
            else:
                with open(f"done_jobs_{prefix}.pkl", "wb") as f:
                    pickle.dump(done_jobs, f)
            save_in_batches(5000, index_type, index, prefix, index_size, unsafe_pickle=False, use_json=use_json, pool=pool)
        except Exception as e:
            print(e)
            print("Pickle Failure")
        gc.collect()
    return index, books_to_index

def merge_parts(parts: list[str], segment_index: str | int=-1, save_dir: str=index_dir):
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

def merge_index(dir: str=index_dir, naming_rule: str=r"part([0-9]+)_inverted_([0-9]+).pkl"):
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