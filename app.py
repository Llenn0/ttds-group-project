import os
import sys
import pickle
import time
import traceback

from flask import Flask, request
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv
from cosine_semantic import SemanticSearch
from google.cloud import storage
from flask_cors import CORS

# Load Environment Variables
load_dotenv()

# Use a service account defined by credentials.json in development and a server environment variable in production
if os.getenv("ENV") == "DEV":
    print("Running in Dev")
    cred = credentials.Certificate('credentials.json')
    fb_app = firebase_admin.initialize_app(cred)
elif os.getenv("ENV") == "PROD":
    print("Running in Production")
    fb_app = firebase_admin.initialize_app()
else:
    print("ERROR IN ENVIRONMENT VARIABLES")
    sys.exit(1)

# Initialise Firestore
db = firestore.client()
coll = db.collection('index')

# Initialise Flask
app = Flask(__name__)
CORS(app)

# Download required files from Cloud Storage
print("Downloading Semantic Embeddings...")
storage_client = storage.Client(project='moonlit-oven-412316')
bucket = storage_client.bucket('ttds-static')
if not os.path.isfile('document_embeddings.pkl'):
    blob = bucket.blob('document_embeddings.pkl')
    blob.download_to_filename("document_embeddings.pkl")
    print("Semantic Embeddings Downloaded.")
else:
    print("Found Semantic Embeddings File")

if not os.path.isfile('KeywordSearch/lookup_table.npz'):
    blob = bucket.blob('lookup_table.npz')
    blob.download_to_filename("KeywordSearch/lookup_table.npz")
    print("Lookup Table Downloaded.")
else:
    print("Found Lookup Table File")

if not os.path.isfile('KeywordSearch/all_tokens.pkl'):
    blob = bucket.blob('all_tokens.pkl')
    blob.download_to_filename("KeywordSearch/all_tokens.pkl")
    print("All Tokens Downloaded.")
else:
    print("Found All Tokens File")

# Initialise and load search modules
searcher = SemanticSearch()
import KeywordSearch.loader as loader

from KeywordSearch.kwsearch import bool_search, adv_search
from KeywordSearch.cloud_index import CloudIndex

inverted_index = CloudIndex(coll, size_limit=5000)
boolean_search_cache = dict()
phrase_search_cache = dict()
paging_cache_limit = 20

def format_book_ids(docIds: list[int], startNum: int, endNum: int, totalNum: int) -> dict:
    if docIds:
        return [{"id": docId, "title": loader.metadata[docId][2], 
                "author": loader.metadata[docId][3], "subject": ", ".join(loader.metadata[docId][1]), 
                "bookshelf": "bookshelf test", "language": ", ".join(loader.metadata[docId][0])} 
                for docId in docIds[startNum:min(endNum, totalNum)]]
    else:
        return []

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/hello')
def hello():
    return 'world!'


@app.route('/semantic', methods=["POST"])
def semantic_search():
    data = request.get_json()

    search = data["query"]
    languages = data["languages"]
    subjects = data["subjects"]
    page = data["page"]
    numPerPage = data["numPerPage"]
    startNum = (page-1) * numPerPage
    endNum = startNum + numPerPage

    start = time.time()
    results = searcher.runSearch(search)
    queryTime = time.time() - start

    results = list(zip(*results))
    docIds, scores = list(results[0]), list(results[1])
    totalNum = len(docIds)

    res_json = {"books": [{"id": int(docId[2:]), "title": "book title", "author": "book author", "subject": "book subject", "bookshelf": "bookshelf test", "language": "English"} for docId in docIds[startNum:endNum]], "queryTime": queryTime, "totalNum": totalNum}
    return res_json

@app.route('/clearcloudindex', methods=["POST"])
def semantic_search():
    err_msg = "No error"
    cloud_index_size = len(inverted_index.cache)
    try:
        data = request.get_json()
        if "clear_cache" in data and data["clear_cache"]:
            inverted_index.clear()
    except Exception as e:
        err_msg = '\n'.join(traceback.format_exception(e))
        print(err_msg)
    
    res_json = {"err_msg" : err_msg, "index_size" : cloud_index_size}
    return res_json

@app.route('/boolean', methods=["POST"])
def boolean_search():
    global boolean_search_cache
    err_msg = "No error"
    try:
        data = request.get_json()
        search_query = data["query"]
        languages = data["languages"]
        subjects = data["subjects"]
        page = data["page"]
        numPerPage = data["numPerPage"]
        max_distance = data["dist"] if "dist" in data else 3
        startNum = (page-1) * numPerPage
        endNum = startNum + numPerPage
        
        if "clear_cache" in data and data["clear_cache"]:
            boolean_search_cache.clear()
        
        start = time.time()
        query_info = search_query + str(sorted(languages)) + str(sorted(subjects)) + str(max_distance)
        docIds = boolean_search_cache.get(query_info, None)
        if docIds is None:
            if len(boolean_search_cache) > paging_cache_limit:
                oldest_result = list(boolean_search_cache.keys())[0]
                del boolean_search_cache[oldest_result]
            try:
                boolean_search_cache[query_info] = sorted(bool_search(search_query, inverted_index, languages, subjects, max_distance))
            except Exception as e:
                err_msg = '\n'.join(traceback.format_exception(e))
                print(err_msg)
                docIds = []
            else:
                docIds = boolean_search_cache[query_info]
        queryTime = time.time() - start
    except Exception as e:
        docIds = []
        startNum = endNum = totalNum = queryTime = -1
        err_msg = '\n'.join(traceback.format_exception(e))

    totalNum = len(docIds)
    res_json = {"books": format_book_ids(docIds, startNum, endNum, totalNum), 
                "queryTime": queryTime, "totalNum": totalNum, "err_msg" : err_msg, 
                "cache_size" : inverted_index.gc()}
    
    return res_json

@app.route('/phrase', methods=["POST"])
def phrase_search():
    global phrase_search_cache
    err_msg = "No error"
    try:
        data = request.get_json()
        search_query = data["query"]
        languages = data["languages"]
        subjects = data["subjects"]
        page = data["page"]
        numPerPage = data["numPerPage"]
        max_distance = data["dist"] if "dist" in data else 3
        startNum = (page-1) * numPerPage
        endNum = startNum + numPerPage

        if "clear_cache" in data and data["clear_cache"]:
            phrase_search_cache.clear()

        start = time.time()
        query_info = search_query + str(sorted(languages)) + str(sorted(subjects)) + str(max_distance)
        docIds = phrase_search_cache.get(query_info, None)
        if docIds is None:
            if len(phrase_search_cache) > paging_cache_limit:
                oldest_result = list(phrase_search_cache.keys())[0]
                del phrase_search_cache[oldest_result]
            try:
                phrase_search_cache[query_info] = sorted(bool_search(search_query, inverted_index, languages, subjects, max_distance, force_phrase=True))
            except Exception as e:
                err_msg = '\n'.join(traceback.format_exception(e))
                print(err_msg)
                docIds = []
            else:
                docIds = phrase_search_cache[query_info]
        queryTime = time.time() - start
    except Exception as e:
        docIds = []
        startNum = endNum = totalNum = queryTime = -1
        err_msg = '\n'.join(traceback.format_exception(e))

    totalNum = len(docIds)
    res_json = {"books": format_book_ids(docIds, startNum, endNum, totalNum), 
                "queryTime": queryTime, "totalNum": totalNum, "err_msg" : err_msg, 
                "cache_size" : inverted_index.gc()}
    return res_json

@app.route('/advanced', methods=["POST"])
def advanced_search():
    err_msg = "No error"
    try:
        data = request.get_json()
        author_query = data["author"]
        title_query = data["title"]
        languages = data["languages"]
        subjects = data["subjects"]
        page = data["page"]
        numPerPage = data["numPerPage"]
        startNum = (page-1) * numPerPage
        endNum = startNum + numPerPage

        start = time.time()
        try:
            docIds = sorted(adv_search(author_query, title_query, languages, subjects))
        except Exception as e:
            err_msg = '\n'.join(traceback.format_exception(e))
            print(err_msg)
            docIds = []
        queryTime = time.time() - start
        totalNum = len(docIds)
    except Exception as e:
        docIds = []
        startNum = endNum = totalNum = queryTime = -1
        err_msg = '\n'.join(traceback.format_exception(e))
    
    res_json = {"books": format_book_ids(docIds, startNum, endNum, totalNum), 
                "queryTime": queryTime, "totalNum": totalNum, "err_msg" : err_msg}
    
    return res_json

@app.route('/category', methods=["POST"])
def display_category():
    err_msg = "No error"
    try:
        data = request.get_json()
        bookshelf_id = data["bookshelf"]
        page = data["page"]
        numPerPage = data["numPerPage"]
        startNum = (page-1) * numPerPage
        endNum = startNum + numPerPage

        start = time.time()
        try:
            docIds = loader.category_dict.get(str(bookshelf_id), [])
        except Exception as e:
            err_msg = '\n'.join(traceback.format_exception(e))
            print(err_msg)
            docIds = []
        queryTime = time.time() - start
        totalNum = len(docIds)
    except Exception as e:
        docIds = []
        startNum = endNum = totalNum = queryTime = -1
        err_msg = '\n'.join(traceback.format_exception(e))

    res_json = {"books": format_book_ids(docIds, startNum, endNum, totalNum), 
                "queryTime": queryTime, "totalNum": totalNum, "err_msg" : err_msg}
    
    return res_json


# @app.route('/getdocs')
# def docs():
#     index = db.collection("index")
#     docs = index.stream()
#     response = ""
#
#     for doc in docs:
#         response += (str(doc.id) + "<br>")
#         for hits in doc.to_dict().values():
#             for hit in hits:
#                 response += (hit + "<br>")
#         response += "<br>"
#     return response


if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
