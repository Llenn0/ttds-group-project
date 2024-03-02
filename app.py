import os
import sys
import pickle
from flask import Flask, request
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv
from run_semantic import SemanticSearch
from google.cloud import storage
from flask_cors import CORS

load_dotenv()

# Use a service account.
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

db = firestore.client()
coll = db.collection('index')

app = Flask(__name__)
CORS(app)

print("Downloading Semantic Embeddings...")
storage_client = storage.Client(project='moonlit-oven-412316')
bucket = storage_client.bucket('ttds-static')
if not os.path.isfile('document_embeddings.pkl'):
    blob = bucket.blob('document_embeddings.pkl')
    blob.download_to_filename("document_embeddings.pkl")
    print("Semantic Embeddings Downloaded.")
else:
    print("Found Semantic Embeddings File")

if not os.path.isfile('lookup_table.npz'):
    blob = bucket.blob('lookup_table.npz')
    blob.download_to_filename("lookup_table.npz")
    print("Lookup Table Downloaded.")
else:
    print("Found Lookup Table File")

if not os.path.isfile('all_tokens.pkl'):
    blob = bucket.blob('all_tokens.pkl')
    blob.download_to_filename("all_tokens.pkl")
    print("All Tokens Downloaded.")
else:
    print("Found All Tokens File")

searcher = SemanticSearch()
from KeywordSearch.kwsearch import bool_search

# Adds files from pickle to the server - ONLY FOR TESTING PURPOSES
def create_index():
    with open("inverted_index.pkl", "rb") as f:
        ebooks = pickle.load(f)
        for i in range(0, len(ebooks)):
            if ebooks[i] != {}:
                data_array = []
                for (k, v) in ebooks[i].items():
                    item_string = f"{str(k)}:"
                    for hit in v:
                        item_string += f" {str(hit)}"
                    data_array.append(item_string)
                data = {"appears_in": data_array}
                print(f"Writing data for doc id {i}")
                coll.document(str(i)).set(data)
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
    results = searcher.runSearch(search)
    results = list(zip(*results))
    docIds, scores = list(results[0]), list(results[1])
    for i in range(len(docIds)):
        docIds[i] = {"id":docIds[i]}
    res_json = {"docIds":docIds}
    return res_json

@app.route('/boolean', methods=["POST"])
def boolean_search():
    data = request.get_json()
    search = data["query"]
    docIds = bool_search(search)

    res_json = {"docIds" : [{"id" : docId} for docId in docIds]}
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
