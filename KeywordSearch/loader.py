import concurrent.futures
import pickle
import os
import gc
import re
import glob
import traceback

from nltk.corpus import stopwords
from Stemmer import Stemmer

from Preprocessing import preprocess_pipeline

VALID_BOOKS_PATH = "valid_books.pkl"
ALL_TOKENS_PATH = "all_tokens.pkl"
LOG_PATH = "kwsearch.log"
print("Please ignore the syntax warnings as small integers in CPython are singletons")
print("Using `is` instead of `=` for comparison in performance-critical code is acceptable")

def init_module():
    global processed_books, raw_dir, token_dir, stopwords_set, stemmer, valid_books, all_tokens, tokens_fetched

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

    if os.path.exists(VALID_BOOKS_PATH):
        with open(VALID_BOOKS_PATH, "rb") as f:
            k_vb, offset_vb, valid_books = pickle.load(f)
    else:
        k_vb = offset_vb = -1
        valid_books = set()
    
    if os.path.exists(ALL_TOKENS_PATH):
        with open(ALL_TOKENS_PATH, "rb") as f:
            k_at, offset_at, all_tokens = pickle.load(f)
    else:
        k_at = offset_at = -1
        all_tokens = tuple()
    
    tokens_fetched = all(val >= 0 for val in (k_vb, offset_vb, k_at, offset_at))

    if os.path.exists(LOG_PATH) and os.path.isfile(LOG_PATH):
        os.remove(LOG_PATH)

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
    with open(fname, 'r', encoding="UTF-8", errors="ignore") as f:
        return set(stemmer.stemWord(token) for token in f.read().splitlines() if token not in stopwords_set)

def load_token_vocab(load_from: str="english_books.txt", k: int=500, offset: int=0) -> tuple:
    try:
        with open("all_tokens.pkl", 'rb') as f:
            k_, offset_, all_tokens = pickle.load(f)
            if k < 0:
                k = k_
            assert k == k_ and offset == offset_
    except:
        all_tokens_set = set()
        if load_from:
            with open(load_from, "rb") as f:
                book_list = [book_id.strip() for book_id in f.readlines()]
            book_list = [int(book_id[2:]) for book_id in book_list if book_id]
        else:
            book_list = [int(filename.split('_')[0][2:]) for filename in glob.glob("PG*.txt", root_dir=token_dir)]
        list_length = len(book_list)
        if offset >= list_length:
            return
        if k < 0:
            k = list_length
        book_list = set(book_list[offset:min(list_length, offset + k)])
        book_path_template = token_dir + "PG%d_tokens.txt"

        complete_counter = 0
        failed_jobs = []
        
        print("Start fetching tokens")
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
                    with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                        f.write(f"Fetch token vocab failure at book {book_id}:\n{''.join(traceback.format_exception(e))}\n")
                    failed_jobs.append(book_id)
                complete_counter += 1

                print(f"Finished fetching tokens in {complete_counter} books...", end="\r")
            print(f"Finished fetching tokens in {complete_counter} books...", flush=True)
        print(f"\n{len(failed_jobs)}/{len(jobs)} token fetching jobs failed", flush=True)

        valid_books = book_list - set(failed_jobs)
        all_tokens = tuple([''] + sorted(all_tokens_set))

        with open("valid_books.pkl", "wb") as f:
            pickle.dump((k, offset, valid_books), f)

        with open("all_tokens.pkl", "wb") as f:
            pickle.dump((k, offset, all_tokens), f)

    return all_tokens

def load_segment(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
        
def load_merged_index(dir: str="index", save_merged: bool=False):
    naming_regex = re.compile(r"([0-9]+)_merged.pkl")
    segments = [[int(naming_regex.fullmatch(filename).group(1)), filename] for filename in glob.glob("*_merged.pkl", root_dir=dir)]
    segments.sort(key=lambda x: x[0])
    
    # Single-threaded version
    # for i, filename in segments:
    #     with open(os.path.join(dir, filename), "rb") as f:
    #         index += list(pickle.load(f))
    #     gc.collect()
    #     print(f"Finished loading segment {i}...", end="\r", flush=True)
    # print("done")

    print(f"{len(segments)} segments to load")
    complete_counter = 0
    index_dict = dict()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        jobs = {pool.submit(load_segment, os.path.join(dir, filename)) : segment_index
                for segment_index, filename in segments}
        for job in concurrent.futures.as_completed(jobs):
            segment_index = jobs[job]
            try:
                index_dict[segment_index] = job.result()
            except Exception as e:
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"Load index failure at segment {segment_index}:\n{''.join(traceback.format_exception(e))}\n")
            complete_counter += 1
            if (complete_counter % 100 is 0):
                gc.collect()
            print(f"Finished loading {complete_counter} segments...", end="\r", flush=True)
    
    index = []
    complete_counter = 0
    for _, segment in sorted(index_dict.items(), key=lambda x: x[0]):
        index.extend(segment)
        complete_counter += 1
        if (complete_counter % 100 is 0):
            gc.collect()
        print(f"Finished merging {complete_counter} segments...", end="\r", flush=True)
    
    del index_dict
    gc.collect()
    print("\nGarbage collection done")

    if save_merged:
        with open(os.path.join(dir, f"full_index.pkl"), "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving done")

    print("All done")
    return index