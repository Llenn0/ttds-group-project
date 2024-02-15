import os
import sys
import pickle
from flask import Flask
import firebase_admin
from firebase_admin import firestore, credentials
from dotenv import load_dotenv

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
