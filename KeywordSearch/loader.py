import os
import gc
import re
import csv
import glob
import time
import pickle
import traceback
import concurrent.futures
from sys import platform
from collections import defaultdict

from nltk.corpus import stopwords
from Stemmer import Stemmer
from tqdm import tqdm
import nltk

from Preprocessing import preprocess_ebooks

isWin = platform.lower() == "win32"
deployment_path = "/app/"

# Paths
LOOKUP_TABLE_PATH = "KeywordSearch/lookup_table.npz"
VALID_BOOKS_PATH = "KeywordSearch/processed_books.pkl"
ALL_TOKENS_PATH = "KeywordSearch/all_tokens.pkl"
LOG_PATH = "kwsearch.log"
index_dir = "index/"
print("Please ignore the syntax warnings as small integers in CPython are singletons")
print("Using `is` instead of `=` for comparison in performance-critical code is acceptable")

if not isWin:
    LOOKUP_TABLE_PATH = deployment_path + LOOKUP_TABLE_PATH
    VALID_BOOKS_PATH = deployment_path + VALID_BOOKS_PATH
    ALL_TOKENS_PATH = deployment_path + ALL_TOKENS_PATH
    index_dir = deployment_path + index_dir

print("Downloading stopwords...")
nltk.download('stopwords')

def init_module():
    global processed_books, raw_dir, token_dir, index_dir
    global stopwords_set, stemmer, processed_books, all_tokens, tokens_fetched, language_code
    global metadata, all_subjects

    try:
        with open(VALID_BOOKS_PATH, 'rb') as f:
            processed_books = set(pickle.load(f))
    except:
        processed_books = set()
    
    try:
        with open("pgpath", 'r', encoding="UTF-8") as f:
            token_dir = f.readline().strip().replace('\\', '/')
        if token_dir[-1] != '/':
            token_dir = token_dir + '/'
    except:
        token_dir = ''

    raw_dir = token_dir.replace("ttds-tokens", "raw")
    stopwords_set = frozenset(stopwords.words("english"))
    stemmer = Stemmer("english")

    if not os.path.exists(token_dir):
        os.makedirs(token_dir)
    
    if os.path.exists(ALL_TOKENS_PATH):
        with open(ALL_TOKENS_PATH, "rb") as f:
            k_at, offset_at, all_tokens = pickle.load(f)
    else:
        k_at = offset_at = -1
        all_tokens = tuple()
    
    tokens_fetched = processed_books and all_tokens

    if os.path.exists(LOG_PATH) and os.path.isfile(LOG_PATH):
        os.remove(LOG_PATH)
    
    with open("language-codes.csv", 'r', encoding="utf-8") as f:
        f.readline()
        reader = csv.reader(f)
        regex_extract = re.compile(r"[^A-Za-z ]")
        tmp = ((code, regex_extract.split(lan)[0].lower()) for code, lan in reader)
        language_code = defaultdict((lambda : "english"), tmp)
    
    metadata, all_subjects = load_lan_dict()

    if not isWin:
        token_dir = deployment_path + token_dir
        raw_dir = deployment_path + raw_dir

def load_lan_dict(path: str="metadata/metadata.csv") -> tuple[defaultdict, dict]:
    if not isWin:
        path = deployment_path + path
    extract_item = re.compile(r"\'(\w+)\'")
    lan_dict = defaultdict(lambda : set())
    sub_dict = dict()
    all_lan = set()
    all_sub = set()
    meta = dict()
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        f.readline()
        reader = csv.reader(f)
        records = [
            (
                int(record[0][2:]), 
                tuple(language_code[lan] for lan in extract_item.findall(record[5])),
                sorted(extract_item.findall(record[7].casefold()))
            ) for record in reader
        ]
    
    for book_id, languages, subjects in records:
        lan_dict[languages].add(book_id)
        all_lan.add(languages)
        all_sub.update(subjects)
    
    # Memory-saving measure: use the same string object for same subject across all books
    all_sub_sorted = sorted(all_sub)
    for book_id, _, subjects in records:
        sub_dict[book_id] = tuple(all_sub_sorted[all_sub_sorted.index(sub)] for sub in subjects)
    
    for lan in all_lan:
        for book_id in lan_dict[lan]:
            meta[book_id] = (lan, sub_dict[book_id])

    return meta, all_sub_sorted

def process_first_k_books(k: int=-1, offset: int=0, batch_size=500):
    global processed_books
    
    book_list = sorted(int(fname.split('_')[0][2:]) for fname in glob.glob("*.txt", root_dir=raw_dir))
    book_list = [book_id for book_id in book_list if (book_id not in processed_books) and (book_id in metadata)]
    list_length = len(book_list)

    if offset >= list_length:
        return
    
    if k > 0:
        book_list = book_list[offset:min(list_length, offset + k)]

    print(f"{len(book_list)} books to process")

    split_by_lan = defaultdict(lambda : list())
    for book_id in book_list:
        split_by_lan[metadata[book_id][0]].append(book_id)

    failed_jobs = []
    completed_jobs = 0

    all_tokens_set = set()

    with concurrent.futures.ProcessPoolExecutor() as pool:
        jobs = dict()
        with tqdm(total=len(book_list), desc="Initialising...") as pbar:
            for lan, lan_book_list in split_by_lan.items():
                pbar.set_description(f"Submitting jobs for language {lan}...")
                if len(lan_book_list) < batch_size:
                    jobs[pool.submit(preprocess_ebooks, lan_book_list, lan, raw_dir, token_dir)] = lan_book_list
                else:
                    list_length = len(lan_book_list)
                    num_batches = list_length // batch_size
                    if list_length % batch_size:
                        num_batches += 1
                    for i in range(num_batches):
                        start = i * batch_size
                        end = min(list_length, start + batch_size)
                        slice = lan_book_list[start:end]
                        jobs[pool.submit(preprocess_ebooks, slice, lan, raw_dir, token_dir)] = slice

            pbar.set_description("Tokenising")
            for job in concurrent.futures.as_completed(jobs):
                books = jobs[job]
                try:
                    all_tokens_set.update(job.result())
                    processed_books.update(books)
                except Exception as e:
                    with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                        f.write(f"Tokenisation failure at {books}:\n{''.join(traceback.format_exception(e))}\n")
                    failed_jobs.extend(books)
                job_size = len(books)
                completed_jobs += job_size
                pbar.update(job_size)

    gc.collect()
    with open("processed_books.pkl", "wb") as f:
        pickle.dump(processed_books, f)

    del processed_books, jobs, split_by_lan

    gc.collect()
    with open("all_tokens.pkl", "wb") as f:
        pickle.dump((k, offset, tuple(sorted(all_tokens_set)) if '' in all_tokens_set else tuple([''] + sorted(all_tokens_set))), f)
    
    print(f"{len(failed_jobs)}/{completed_jobs} pre-processing jobs failed:\n- " + '\n- '.join(str(i) for i in failed_jobs))

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

def load_segment(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)

def load_dummy_segment(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return [{k : v.shape[0] for k, v in token_dict.items()} for token_dict in pickle.load(f)]
        
def load_merged_index(dir_: str=index_dir, save_merged: bool=False, max_workers: int=None, dummy: bool=False):
    if not isWin:
        dir_ = deployment_path + dir_
    start_time = time.time()

    load_func = load_dummy_segment if dummy else load_segment
    naming_regex = re.compile(r"([0-9]+)_merged.pkl")
    segments = [[int(naming_regex.fullmatch(filename).group(1)), filename] for filename in glob.glob("*_merged.pkl", root_dir=dir_)]
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        jobs = {pool.submit(load_func, os.path.join(dir_, filename)) : segment_index
                for segment_index, filename in segments}
        for job in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs), desc="Loading segments"):
            segment_index = jobs[job]
            try:
                index_dict[segment_index] = job.result()
            except Exception as e:
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"Load index failure at segment {segment_index}:\n{''.join(traceback.format_exception(e))}\n")
            complete_counter += 1
            if (complete_counter % 100 == 0):
                gc.collect()
            # print(f"Finished loading {complete_counter} segments...", end="\r", flush=True)
    
    index = []
    complete_counter = 0
    for _, segment in tqdm(sorted(index_dict.items(), key=lambda x: x[0]), total=len(index_dict), desc="Merging segments"):
        index.extend(segment)
        complete_counter += 1
        if (complete_counter % 100 == 0):
            gc.collect()
        # print(f"Finished merging {complete_counter} segments...", end="\r", flush=True)
    
    del index_dict
    gc.collect()
    run_time = int(time.time() - start_time)
    h = run_time // 3600
    h_str = f"{h} hour{'s' if h > 1 else ''} " if h > 0 else ''
    run_time %= 3600
    m = run_time // 60
    m_str = f"{m} minute{'s' if m > 1 else ''}  " if m > 0 else ''
    run_time %= 60
    print(f"\nGarbage collection done")
    print(f"The index took {h_str}{m_str}{run_time} seconds to load")

    if save_merged:
        with open(os.path.join(dir_, f"full_index.pkl"), "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saving done")

    print("All done")
    return index