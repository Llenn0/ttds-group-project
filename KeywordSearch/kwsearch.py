import pickle
import re

import numpy as np
import scipy.sparse

from KeywordSearch import indexing, utils
from KeywordSearch.loader import stemmer, load_token_vocab

with open("valid_books.pkl", "rb") as f:
    _, _, valid_books = pickle.load(f)

all_tokens = load_token_vocab(k=-1)
token_index_dict: dict[str, int] = indexing.ZeroDict((token, i) for i, token in enumerate(all_tokens))
all_tokens_set: set[str] = set(all_tokens)
book_index: np.ndarray[int]
book_index, _ = utils.cast2intarr(np.array(sorted(valid_books)), delta_encode=False)

regex_phrase = re.compile(r"\"[\w\s]\"")
regex_non_alnum = re.compile(r"[^A-Za-z0-9.]")
regex_bracket = re.compile(r"\((.+)\)( +(?:NOT|AND|OR))?")
regex_bool_op = re.compile(r"(NOT|AND|OR)")

lookup_table: scipy.sparse.csr_matrix = scipy.sparse.load_npz("lookup_table.npz")

operators = frozenset(("NOT", "AND", "OR"))

all_elems_list = list(range(lookup_table.shape[1]))
all_elems_set = frozenset(all_elems_list)
all_elems_arr = np.array(all_elems_list, dtype=np.int32)
del all_elems_list

def bool_search(query: str, debug: bool=False) -> set:
    if debug:
        print(regex_bracket.split(query))
    tokens = (bool_search_atomic(token, debug) for token in regex_bracket.split(query) if token)
    is_not = is_and = is_or = False
    not_first = False
    valid, (is_not, is_and, is_or) = next(tokens)
    for token_eval, (is_not_, is_and_, is_or_) in tokens:
        is_not |= is_not_; is_and |= is_and_; is_or |= is_or_
        if isinstance(token_eval, list):
            continue
        # print(f"Parse: {len(token_eval)} {'OR' if is_or else ''} {'AND' if is_and else ''} {'NOT' if is_not else ''}")
        if is_or:
            if is_not:
                if not_first:
                    valid = token_eval | (all_elems_set - valid) 
                else:
                    valid |= (all_elems_set - token_eval)
                is_not = False
            else:
                valid |= token_eval
            is_or = is_not = False
        elif is_and:
            if is_not:
                if not_first:
                    valid = token_eval - valid
                else:
                    valid -= token_eval
                is_not = False
            else:
                valid &= token_eval
            is_and = is_not = False
        elif is_not:
            valid -= token_eval
        else:
            print("Grammar error?")
    return valid, (is_not, is_and, is_or)

def bool_search_atomic(query: str, debug: bool) -> set:
    if '(' in query:
        return bool_search(query, debug)
    query = query.strip()
    if not query:
        return set(), (False, False, False)
    tokens = [token for token in regex_bool_op.split(query) if token]
    is_not = is_and = is_or = False
    not_first = False
    valid = []
    for token in tokens:
        if token in operators:
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
                        is_not = False
                    valid = np.union1d(valid, token_eval)
                    
                    is_or = False
                    if not valid.shape: return set(), (is_not, is_and, is_or)
                elif is_and:
                    if is_not:
                        if not_first:
                            valid = np.setdiff1d(token_eval, valid, assume_unique=True)
                        else:
                            valid = np.setdiff1d(valid, token_eval, assume_unique=True)
                        is_not = False
                    else:
                        valid = np.intersect1d(valid, token_eval, assume_unique=True)
                    
                    is_and = False
                    if not valid.shape: return set(), (is_not, is_and, is_or)
                else:
                    valid = token_eval
            else:
                # unseen word or stopword
                is_not = is_and = is_or = False
    if isinstance(valid, np.ndarray):
        valid = set(valid.tolist())
    return valid, (is_not, is_and, is_or)

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