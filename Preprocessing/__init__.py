import nltk
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from unidecode import unidecode


def unpickle_file(name):
    """
    :param name: str  |  Only 'bookshelf_categories', 'ebooks_by_bookshelf'
    """
    with open(name + '.pkl', 'rb') as file_in:
        unpickled_file = pickle.load(file_in)

    with open(name + '.txt', 'w') as file_out:
        for key, value in unpickled_file.items():
            file_out.write(f'{key}: {value}\n')

def trim_ebook(book_id, raw_dir: str='', trim_dir: str=''):
    """
    :param name: str  |  e.g. 'PG10000'
    """
    with open(raw_dir + book_id + '.txt', 'r') as file_in:
        contents = file_in.read()
    
    title = re.search(r'Title: (.*)', contents).group(1).upper()
    start_pattern = f'\*\*\* START OF THE PROJECT GUTENBERG EBOOK {title} \*\*\*'
    end_pattern = f'\*\*\* END OF THE PROJECT GUTENBERG EBOOK {title} \*\*\*'

    start_index = re.search(start_pattern, contents).start() + len(start_pattern) - 6
    end_index = re.search(end_pattern, contents).end() - len(end_pattern) + 6

    # Blank lines removed here for readability; not optimal in terms of efficiency
    trimmed_contents = re.sub(r'\n{3,}', '\n\n', contents[start_index:end_index])

    with open(trim_dir + book_id + '_trimmed.txt', 'w') as file_out:
        file_out.write(trimmed_contents)

def preprocess_ebook(book_id, stopwords, stemmer, numeric=False, raw_dir: str='', 
                     trim_dir: str='', data_dir: str=''):
    with open(trim_dir + book_id + '_trimmed.txt', 'r') as file_in:
        contents = file_in.read()
    
    convert_to_ascii = unidecode(contents)

    # Based on :param numeric, match alphabetic or alphanumeric characters
    regex = r'a-z0-9' if numeric else r'a-z'

    # Remove all special characters except apostrophes in contractions (to match stopword removal)
    pattern1 = f'[^{regex}\' ]|(?<![{regex}])\'|\'(?![{regex}])'
    tokens = re.sub(pattern1, ' ', convert_to_ascii.lower()).split()
    
    terms = [stemmer.stem(word) for word in tokens if word not in stopwords]

    with open(data_dir + book_id + '_processed.txt', 'w') as file_out:
        file_out.write(' '.join(terms))


if __name__ == '__main__':
    # unpickle_file('bookshelf_categories')
    # unpickle_file('ebooks_by_bookshelf')

    # nltk.download('stopwords')
    stopwords_set = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')

    trim_ebook('PG10000')
    preprocess_ebook('PG10000', stopwords_set, stemmer)
