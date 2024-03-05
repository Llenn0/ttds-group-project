import os
import sys
import pickle
import time

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

loader.init_module()
from KeywordSearch.kwsearch import bool_search
from KeywordSearch.cloud_index import CloudIndex

inverted_index = CloudIndex(coll, size_limit=10000)

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

    res_json = {"books": [{"id": docId, "title": "book title", "author": "book author", "subject": "book subject", "bookshelf": "bookshelf test", "language": "English"} for docId in docIds[startNum:endNum]], "queryTime": queryTime, "totalNum": totalNum}
    return res_json


@app.route('/boolean', methods=["POST"])
def boolean_search():
    data = request.get_json()

    search = data["query"]
    languages = data["languages"]
    subjects = data["subjects"]
    page = data["page"]
    numPerPage = data["numPerPage"]
    startNum = (page-1) * numPerPage
    endNum = startNum + numPerPage

    start = time.time()
    docIds = bool_search(search, inverted_index, languages, subjects)
    queryTime = time.time() - start

    inverted_index.gc()
    totalNum = len(docIds)
    res_json = {"books": [{"id": "PG" + str(docId), "title": loader.metadata[docId][2], 
                           "author": loader.metadata[docId][3], "subject": ", ".join(loader.metadata[docId][1]), 
                           "bookshelf": "bookshelf test", "language": ", ".join(loader.metadata[docId][0])} 
                           for docId in docIds[startNum:min(endNum, len(docIds))]], "queryTime": queryTime, "totalNum": totalNum}
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
