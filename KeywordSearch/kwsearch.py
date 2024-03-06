from typing import Iterable

import re
import gc

import numpy as np
import scipy.sparse
from unidecode import unidecode

from KeywordSearch import utils
from KeywordSearch.cloud_index import CloudIndex
from KeywordSearch.loader import stemmer, processed_books, all_tokens, stopwords_dict, all_lan_single, sub_dict, LOOKUP_TABLE_PATH
from KeywordSearch.loader import subject_index, title_index, author_index, subject_ids, title_ids, author_ids

# Sparse look-up table aids boolean search
lookup_table: scipy.sparse.csr_array | None
try:
    lookup_table = scipy.sparse.load_npz(LOOKUP_TABLE_PATH)
except:
    lookup_table = None

# Pre-compiled regular expressions for parsing boolean queries
regex_phrase: re.Pattern                = re.compile(r"\"[\w\s]\"")
regex_non_alnum: re.Pattern             = re.compile(r"[^A-Za-z0-9.]")
regex_bracket: re.Pattern               = re.compile(r"\((.+)\)( +(?:NOT|AND|OR))?")
regex_bool_op: re.Pattern               = re.compile(r"(NOT|AND|OR)")
regex_tokenise: re.Pattern              = re.compile(r"\b\w+\b")

# Frozen set of boolean search operators
bool_ops = frozenset(("NOT", "AND", "OR"))

# Other global variables that are expensive to generate
token_index_dict: dict[str, int]        = utils.ZeroDict((token, i) for i, token in enumerate(all_tokens))
all_tokens_set: set[str]                = set(all_tokens)
book_index: np.ndarray[int]             = utils.cast2intarr(np.array(sorted(processed_books)), delta_encode=False)[0]
all_elems_set: set[int]                 = frozenset(processed_books)
all_elems_arr: np.ndarray[np.int32]     = np.array(sorted(processed_books), dtype=np.int32)

# Garbage collection
gc.collect()

def update_index(processed_books, all_tokens):
    global token_index_dict, all_tokens_set, book_index, all_elems_set, all_elems_arr
    token_index_dict = utils.ZeroDict((token, i) for i, token in enumerate(all_tokens))
    all_tokens_set = set(all_tokens)
    book_index = utils.cast2intarr(np.array(sorted(processed_books)), delta_encode=False)[0]
    all_elems_set = frozenset(range(lookup_table.shape[1]))
    all_elems_arr = np.arange(lookup_table.shape[1], dtype=np.int32)
    gc.collect()

def bool_search(query: str, index: Iterable[dict], lans, subs, debug: bool=False) -> set[int]:
    query = unidecode(query)
    # all_elem = set()
    # for lan in lans:
    #     all_elem.update(lan_dict[lan])
    # for sub in subs:
    #     all_elem.update(sub_dict[sub])
    filtered_results = filter_by_lan_sub(lans, subs)
    current_stopwords = set()
    for lan in lans:
        if lan in stopwords_dict:
            current_stopwords.update(stopwords_dict[lan])
    if regex_bool_op.search(query) is None:
        query_tokens = [word for word in regex_tokenise.findall(query.lower()) if word not in current_stopwords]
        return phrase_search_cloud(query_tokens, index, 3, filter=filtered_results, debug=debug)
    else:
        filter_arr = np.array(list(filtered_results), dtype=np.uint32)
        return _bool_search(query, filter=filtered_results, filter_arr=filter_arr, debug=debug)[0] & filtered_results

def _bool_search(query: str, filter: set[int]=all_elems_set, filter_arr: np.ndarray[int]=all_elems_arr, 
                 debug: bool=False) -> set[int]:
    if debug:
        print(regex_bracket.split(query))
    tokens = (bool_search_atomic(token, filter, filter_arr, debug) for token in regex_bracket.split(query) if token)
    is_not = is_and = is_or = False
    valid, (is_not, is_and, is_or) = next(tokens)
    for token_eval, (is_not_, is_and_, is_or_) in tokens:
        if isinstance(token_eval, list):
            is_not = is_not_; is_and = is_and_; is_or = is_or_
            continue
        if is_or:
            if is_not:
                valid |= (filter - token_eval)
            else:
                valid |= token_eval
            is_or = is_not = False
        elif is_and:
            if is_not:
                valid -= token_eval
            else:
                valid &= token_eval
            is_and = is_not = False
        elif is_not:
            valid = filter - token_eval
        else:
            print(f"Grammar error?")
        
        is_not = is_not_; is_and = is_and_; is_or = is_or_
    return valid, (is_not, is_and, is_or)

def bool_search_atomic(query: str, filter_: set[int]=all_elems_set, filter_arr: np.ndarray[int]=all_elems_arr, debug: bool=False) -> set[int]:
    if '(' in query:
        return _bool_search(query, filter_, filter_arr, debug)
    query = query.strip()
    if not query:
        return set(), (False, False, False)
    tokens = [token for token in (token.strip() for token in regex_bool_op.split(query)) if token]
    is_not = is_and = is_or = False
    not_first = False
    valid = []
    for token in tokens:
        if token in bool_ops:
            if token == "OR":
                is_or = True
                if is_not: not_first = True
            elif token == "NOT":
                is_not = True
                not_first = False
            elif token == "AND":
                is_and = True
                if is_not: not_first = True
        else:
            token_id = token_index_dict[stemmer.stemWord(token.strip().casefold())]
            if token_id:
                token_eval = lookup_table[token_id, :].indices
                if is_or:
                    if is_not:
                        if not_first:
                            valid = np.setdiff1d(filter_arr, valid, assume_unique=True)
                        else:
                            token_eval = np.setdiff1d(filter_arr, token_eval, assume_unique=True)
                    valid = np.union1d(valid, token_eval)
                    
                    is_or = is_not = False
                    if not valid.shape: return set(), (is_not, is_and, is_or)
                elif is_and:
                    if is_not:
                        if not_first:
                            valid = np.setdiff1d(token_eval, valid, assume_unique=True)
                        else:
                            valid = np.setdiff1d(valid, token_eval, assume_unique=True)
                    else:
                        valid = np.intersect1d(valid, token_eval, assume_unique=True)
                    
                    is_and = is_not = False
                    if not valid.shape: return set(), (is_not, is_and, is_or)
                else:
                    valid = token_eval
            else:
                # unseen word or stopword
                is_not = is_and = is_or = False
    if isinstance(valid, np.ndarray):
        valid = set(valid.tolist())
    return valid, (is_not, is_and, is_or)

def phrase_search_cloud(words: list[str], index: CloudIndex, max_dist: int=1, filter: set=all_elems_set, debug: bool=False) -> set[int]:
    search_result = []
    if debug:
        print(stemmer.stemWords(words))
    raw_word_ids = (token_index_dict[stemmer.stemWord(word)] for word in words)
    raw_word_ids = tuple(word_id for word_id in raw_word_ids if word_id)
    word_ids = set(raw_word_ids)
    
    num_words = len(raw_word_ids)
    if num_words == 0:
        return all_elems_set
    elif num_words == 1:
        return set(lookup_table[raw_word_ids[0], :].indices.tolist())
    
    all_ids = list(word_ids)
    intersection = lookup_table[all_ids.pop(), :].indices
    for token_id in all_ids:
        intersection = np.intersect1d(intersection, lookup_table[token_id, :].indices, assume_unique=True)
    intersection = np.intersect1d(intersection, list(filter), assume_unique=True)
    index.preallocate(intersection)

    num_words = len(raw_word_ids)
    if num_words > 30:
        index_entries = []
        num_batches = num_words // 30
        if num_words % 30:
            num_batches += 1
        for i in range(num_batches):
            start = i * 30
            end = min(num_words, start + 30)
            index_entries += index[raw_word_ids[start:end]]
    else:
        index_entries = index[raw_word_ids]

    for docID in intersection:
        occurs = (entry[docID] for entry in index_entries) # use generator to avoid wasting time on non-matches
        first = next(occurs)
        second = next(occurs)
        i = np.searchsorted(first, second, side="left")
        i[i==0] = 1
        matches: np.ndarray = second[(second - first[i-1]) <= max_dist]
        del first, second

        if matches.shape[0]:
            for entry in occurs:
                i = np.searchsorted(matches, entry, side="right")
                i[i==0] = 1
                matches = entry[(entry - matches[i-1]) <= max_dist]
                if not matches.any():
                    break
            else:
                search_result.append(docID)
    return set(search_result)

def filter_by_lan_sub(languages: list[str]|str, subjects: list[str]) -> set[int]:
    lan_result = set()
    if isinstance(languages, str):
        lan_result.update(all_lan_single.get(languages, []))
    else:
        for lan in languages:
            lan_result.update(all_lan_single.get(lan, []))
    sub_result = set()
    for sub in subjects:
        sub_result.update(sub_dict.get(sub, []))

    # Internal conversion of empty set to boolean False or non-empty set to boolean True
    # is faster than length comparisons or bool() casting in most cases
    lan_empty = not lan_result
    sub_empty = not sub_result
    if lan_empty and sub_empty:
        return all_elems_set
    elif lan_empty:
        return sub_result
    elif sub_empty:
        return lan_result
    return sub_result & lan_result

def field_search(tokens: set[str], id_dict: dict[str, int], field_index: scipy.sparse.csr_array):
    author_results = set()
    for token in tokens:
        if token in id_dict:
            author_results.update(field_index[id_dict[token], :].indices.tolist())
    return author_results

def adv_search(author_query: str, title_query: str, languages: list[str], subjects: list[str]) -> set[int]|frozenset[int]:
    result = all_elems_set.copy()

    author_tokens = set(stemmer.stemWords(regex_tokenise.findall(unidecode(author_query).casefold())))
    title_tokens = set(stemmer.stemWords(regex_tokenise.findall(unidecode(title_query).casefold())))
    
    author_results = field_search(author_tokens, author_ids, author_index)
    title_results = field_search(title_tokens, title_ids, title_index)
    lan_sub_results = filter_by_lan_sub(languages, subjects)
    
    for r in (author_results, title_results, lan_sub_results):
        if r:
            result &= r
    
    return result

# [Deprecated] local index version of phrase search
# def phrase_search(words: list[str], index: Iterable[dict], debug: bool=False):
#     search_result = []
#     if debug:
#         print([stemmer.stemWord(word) for word in words if word not in stopwords_set])
#     word_ids = (token_index_dict[stemmer.stemWord(word)] for word in words)
#     word_ids = [word_id for word_id in word_ids if word_id]
#     index_entries = [index[i] for i in word_ids]
#     first = list(set(word_ids))
#     intersection = lookup_table[first.pop(), :].indices
#     for token_id in first:
#         intersection = np.intersect1d(intersection, lookup_table[token_id, :].indices, assume_unique=True)
#     del first, word_ids

#     for docID in intersection:
#         occurs = (entry[docID] for entry in index_entries) # use generator to avoid wasting time on non-matches
#         first = next(occurs)
#         second = next(occurs)
#         matches: np.ndarray = second[first[np.searchsorted(first, second, side="right")-1] == second-1]
#         del first, second

#         if matches.shape[0]:
#             for entry in occurs:
#                 i = np.searchsorted(matches, entry, side="right")
#                 i[i==0] = 1
#                 matches = entry[matches[i-1] == entry-1]
#                 if not matches.any():
#                     break
#             else:
#                 search_result.append(docID)
#     return set(search_result)

if __name__ == "__main__":
    print(len(_bool_search("fine AND (okay AND good) AND happy")))