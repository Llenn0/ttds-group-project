import concurrent.futures
import pickle
import os
import glob
import traceback

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
                    with open("log", 'a', encoding="UTF-8") as f:
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

def get_all_tokens(path: str="all_tokens.pkl"):
    with open(path, "rb") as f:
        _, _, all_tokens = pickle.load(f)
    return tuple([''] + list(all_tokens))