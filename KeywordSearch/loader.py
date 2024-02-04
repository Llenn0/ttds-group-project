import concurrent.futures
import pickle
import os

import numpy as np
from nltk.corpus import stopwords
from Stemmer import Stemmer

from Preprocessing import preprocess_pipeline

def init_module():
    global processed_books, raw_dir, token_dir, stopwords_set, stemmer

    try:
        with open("processed_books.pkl", 'rb') as f:
            processed_books = pickle.load(f)
    except:
        processed_books = set()
    
    try:
        with open("pgpath", 'r', encoding="UTF-8") as f:
            token_dir = f.readline().strip()
        if token_dir[-1] not in ('/', '\\'):
            token_dir = token_dir + '/'
    except:
        token_dir = ''

    raw_dir = token_dir.replace("tokens", "raw")
    stopwords_set = frozenset(stopwords.words("english"))
    stemmer = Stemmer("english")

def process_first_k_books(load_from: str="english_books.txt", k: int=500, offset: int=0):
    global processed_books

    with open(load_from, 'r') as f:
        book_list = [book_id.strip() for book_id in f.readlines()]
    book_list = [book_id.upper() for book_id in book_list if book_id]
    list_length = len(book_list)
    if offset >= list_length:
        return
    book_list = set(book_list[offset:min(list_length, offset + k)]) - processed_books
    if not os.path.exists("processed"):
        os.mkdir("processed")
    if not os.path.exists("trimmed"):
        os.mkdir("trimmed")

    print(f"{len(book_list)} books to process")
    failed_jobs = []
    completed_jobs = 0
    with concurrent.futures.ThreadPoolExecutor() as pool:
        jobs = {
            pool.submit(
                preprocess_pipeline, book_id, stopwords_set, stemmer, raw_dir=raw_dir,
                trim_dir="trimmed/", data_dir="processed/")
                : book_id for book_id in book_list
            }
        
        for job in concurrent.futures.as_completed(jobs):
            book_id = jobs[job]
            try:
                job.result()
                processed_books.add(book_id)
            except Exception as e:
                print(e)
                failed_jobs.append(book_id)
            completed_jobs += 1
            print(f"Processed {completed_jobs} books...", end="\r")
    
    with open("processed_books.pkl", 'wb') as f:
        pickle.dump(processed_books, f)
    
    print(f"{len(failed_jobs)}/{len(jobs)} pre-processing jobs failed:\n- " + '\n- '.join(failed_jobs))

def fetch_token_vocab(fname: str, stemmer: Stemmer):
    with open(fname, 'r') as f:
        tokens = [stemmer.stemWord(token) for token in f.read().splitlines() 
                  if token not in stopwords_set]
    vocab = set(tokens)
    return vocab

def build_book_index(fname: str, book_id: int, stemmer: Stemmer, token_index_dict: dict, index: tuple[dict]):
    # todo: variable dtype
    with open(fname, 'r') as f:
        tokens = [token_index_dict[stemmer.stemWord(token)] for token in f.read().splitlines() 
                  if token not in stopwords_set]
    vocab = set(tokens)
    token_num = len(tokens)
    tokens_arr = np.array(tokens, dtype=np.uint16)
    for token in vocab:
        token_occurences = np.where(tokens_arr == token)[0]
        # next_token_pos = token_occurences + 1
        # out_of_bound = next_token_pos >= token_num
        # next_token_pos[out_of_bound] = 0
        # next_tokens = tokens_arr[next_token_pos]
        # next_tokens[out_of_bound] = 0
        # index[token][book_id] = (token_occurences.astype(np.uint16), next_tokens)
        index[token][book_id] = token_occurences.astype(np.uint16)

def build_full_index():
    with open("valid_books.pkl", "rb") as f:
        _, _, valid_books = pickle.load(f)
    with open("all_tokens.pkl", "rb") as f:
        _, _, all_tokens = pickle.load(f)
    all_tokens.insert(0, '') # dummy token
    token_index_dict = dict((token, i) for i, token in enumerate(all_tokens))
    index = tuple(dict() for _ in all_tokens)
    book_path_template = token_dir + "PG%d_tokens.txt"

    complete_counter = 0
    failed_jobs = []
    with concurrent.futures.ThreadPoolExecutor() as pool:
        jobs = {
            pool.submit(
                build_book_index, book_path_template % book_id, book_id, stemmer, token_index_dict, index)
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

def load_token_vocab(load_from: str="english_books.txt", k: int=500, offset: int=0):
    try:
        with open("all_tokens.pkl", 'rb') as f:
            k_, offset_, all_tokens = pickle.load(f)
            assert k == k_ and offset == offset_
    except:
        all_tokens_set = set()
        with open(load_from, "rb") as f:
            book_list = [book_id.strip() for book_id in f.readlines()]
        book_list = [int(book_id[2:]) for book_id in book_list if book_id]
        list_length = len(book_list)
        if offset >= list_length:
            return
        book_list = set(book_list[offset:min(list_length, offset + k)])
        book_path_template = token_dir + "PG%d_tokens.txt"

        complete_counter = 0
        failed_jobs = []
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            jobs = {
                pool.submit(
                    fetch_token_vocab, book_path_template % book_id, stemmer)
                    : book_id for book_id in book_list
                }
            
            for job in concurrent.futures.as_completed(jobs):
                book_id = jobs[job]
                try:
                    all_tokens_set.update(job.result())
                except Exception as e:
                    # raise e
                    failed_jobs.append(book_id)
                complete_counter += 1
                print(f"Finished fetching tokens in {complete_counter} books...", end="\r")
        
        print(f"\n{len(failed_jobs)}/{len(jobs)} token fetching jobs failed")

        valid_books = book_list - set(failed_jobs)
        all_tokens = sorted(all_tokens_set)

        with open("valid_books.pkl", "wb") as f:
            pickle.dump((k, offset, valid_books), f)

        with open("all_tokens.pkl", "wb") as f:
            pickle.dump((k, offset, all_tokens), f)

    return all_tokens

