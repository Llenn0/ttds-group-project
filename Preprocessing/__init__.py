import re

from nltk.corpus import stopwords
from unidecode import unidecode, unidecode_expect_nonascii
from Stemmer import Stemmer, algorithms

from SPGC.cleanup import strip_headers

stemmable_languages = set(algorithms())
stopwords_available = {'hebrew', 'dutch', 'bengali', 'french', 'arabic', 'romanian', 'kazakh', 'italian', 'german', 'indonesian', 'nepali', 'danish', 'english', 'hinglish', 'spanish', 'basque', 'turkish', 'greek', 'slovene', 'chinese', 'russian', 'finnish', 'portuguese', 'norwegian', 'README', 'azerbaijani', 'catalan', 'tajik', 'hungarian', 'swedish'}
stemmer_available = set(algorithms())

def init_preprocessor(languages: tuple[str]=("english",)) -> tuple[Stemmer, frozenset[str]]:
    stemmable = [lan for lan in languages if lan in stemmer_available]
    stoppable = [lan for lan in languages if lan in stopwords_available]

    if "english" in languages:
        stem_lan = "english"
    else:
        stem_lan = stemmable[0] if stemmable else "english"
        if len(stoppable) < 1:
            stoppable = ["english"]

    stemmer = Stemmer("english") if stem_lan not in ("chinese", "japanese") else None # use only English for now
    stopwords_set = set()

    for lan in stoppable:
        addition = stopwords.words(lan)
        if lan in ("chinese", "japanese"):
            tmp = '\n'.join(addition)
            addition = set(unidecode_expect_nonascii(tmp).lower().splitlines())
            del tmp
        stopwords_set.update(addition)

    return stemmer, frozenset(stopwords_set)

def preprocess_ebooks(job_infos: list[int], languages: tuple[str], raw_dir: str='', token_dir: str=''):
    tokens_set = set()
    raw_text_dir = raw_dir + "PG%d_raw.txt"
    tokenised_text_dir = token_dir + "PG%d_tokens.txt"
    regex_brackets = re.compile(r"\[.*\]")
    regex_tokenise = re.compile(r"\b\w+\b")
    stemmer, stopwords_set = init_preprocessor(languages)

    for book_id in job_infos:
        with open(raw_text_dir % book_id, 'r', encoding="UTF-8", errors="ignore") as f:
            contents = strip_headers(f.read()) # strip project gutenberg headers
        
        contents = unidecode(contents, replace_str=' ').lower()
        contents = regex_brackets.sub(' ', contents)
        if stemmer is None:
            tokens = [token for token in regex_tokenise.findall(contents) if token not in stopwords_set]
        else:
            tokens = [stemmer.stemWord(token) for token in regex_tokenise.findall(contents) if token not in stopwords_set]
        
        with open(tokenised_text_dir % book_id, 'w', encoding="UTF-8") as f:
            f.write('\n'.join(tokens))
        
        tokens_set.update(tokens)
    
    return tokens_set
