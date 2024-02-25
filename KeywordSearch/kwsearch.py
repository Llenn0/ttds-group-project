import re
import gc

import numpy as np
import scipy.sparse

from KeywordSearch import utils
from KeywordSearch.loader import stemmer, valid_books, all_tokens, stopwords_set

# Path for look-up table for boolean search
LOOKUP_TABLE_PATH = "lookup_table.npz"


# Sparse look-up table aids boolean search
lookup_table: scipy.sparse.csr_matrix | None
if valid_books and all_tokens:
    lookup_table = scipy.sparse.load_npz(LOOKUP_TABLE_PATH)
else:
    lookup_table = None

# Pre-compiled regular expressions for parsing boolean queries
regex_phrase: re.Pattern                = re.compile(r"\"[\w\s]\"")
regex_non_alnum: re.Pattern             = re.compile(r"[^A-Za-z0-9.]")
regex_bracket: re.Pattern               = re.compile(r"\((.+)\)( +(?:NOT|AND|OR))?")
regex_bool_op: re.Pattern               = re.compile(r"(NOT|AND|OR)")

# Frozen set of boolean search operators
bool_ops = frozenset(("NOT", "AND", "OR"))

# Other global variables that are expensive to generate
token_index_dict: dict[str, int]        = utils.ZeroDict((token, i) for i, token in enumerate(all_tokens))
all_tokens_set: set[str]                = set(all_tokens)
book_index: np.ndarray[int]             = utils.cast2intarr(np.array(sorted(valid_books)), delta_encode=False)[0]
all_elems_set: set[int]                 = frozenset(valid_books)
all_elems_arr: np.ndarray[np.int32]     = np.array(sorted(valid_books), dtype=np.int32)

# Garbage collection
gc.collect()

def update_index(valid_books, all_tokens):
    global token_index_dict, all_tokens_set, book_index, all_elems_set, all_elems_arr
    token_index_dict = utils.ZeroDict((token, i) for i, token in enumerate(all_tokens))
    all_tokens_set = set(all_tokens)
    book_index = utils.cast2intarr(np.array(sorted(valid_books)), delta_encode=False)[0]
    all_elems_set = frozenset(range(lookup_table.shape[1]))
    all_elems_arr = np.arange(lookup_table.shape[1], dtype=np.int32)
    gc.collect()

def bool_search(query: str, debug: bool=False) -> set:
    if debug:
        print(regex_bracket.split(query))
    tokens = (bool_search_atomic(token, debug) for token in regex_bracket.split(query) if token)
    is_not = is_and = is_or = False
    valid, (is_not, is_and, is_or) = next(tokens)
    for token_eval, (is_not_, is_and_, is_or_) in tokens:
        if isinstance(token_eval, list):
            is_not = is_not_; is_and = is_and_; is_or = is_or_
            continue
        if is_or:
            if is_not:
                valid |= (all_elems_set - token_eval)
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
            valid = all_elems_set - token_eval
        else:
            print(f"Grammar error?")
        
        is_not = is_not_; is_and = is_and_; is_or = is_or_
    return valid, (is_not, is_and, is_or)

def bool_search_atomic(query: str, debug: bool) -> set:
    if '(' in query:
        return bool_search(query, debug)
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
                            valid = np.setdiff1d(all_elems_arr, valid, assume_unique=True)
                        else:
                            token_eval = np.setdiff1d(all_elems_arr, token_eval, assume_unique=True)
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

def phrase_search(words: list[str], index: list[dict] | tuple[dict], debug: bool=False):
    search_result = []
    if debug:
        print([stemmer.stemWord(word) for word in words if word not in stopwords_set])
    word_ids = [token_index_dict[stemmer.stemWord(word)] for word in words if word not in stopwords_set]
    index_entries = [index[i] for i in word_ids]
    first = list(set(word_ids))
    intersection = lookup_table[first.pop(), :].indices
    for token_id in first:
        intersection = np.intersect1d(intersection, lookup_table[token_id, :].indices, assume_unique=True)
    del first, word_ids

    for docID in intersection:
        occurs = (entry[docID] for entry in index_entries) # use generator to avoid wasting time on non-matches
        first = next(occurs)
        second = next(occurs)
        matches: np.ndarray = second[first[np.searchsorted(first, second, side="right")-1] == second-1]
        del first, second

        if matches.shape[0]:
            for entry in occurs:
                i = np.searchsorted(matches, entry, side="right")
                i[i==0] = 1
                matches = entry[matches[i-1] == entry-1]
                if not matches.any():
                    break
            else:
                search_result.append(docID)
    return set(search_result)

# def search(query: str):
#     # phrases = regex_phrase.finditer(query)
#     print([token_index_dict[stemmer.stemWord(token)] for token in regex_non_alnum.split(query.casefold()) if token not in indexing.stopwords_set])
#     query_tokens = (token_index_dict[stemmer.stemWord(token)] for token in regex_non_alnum.split(query.casefold()) if token not in indexing.stopwords_set)
#     token = next(query_tokens)
#     _, valid = np.nonzero(lookup_table[token, :])
#     for token in query_tokens:
#         if token:
#             _, valid = np.nonzero(lookup_table[token, valid])
#             print(token, all_tokens[token], valid)
#     return valid

if __name__ == "__main__":
    print(len(bool_search("fine AND (okay AND good) AND happy")))