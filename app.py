import os
from flask import Flask
import firebase_admin
from firebase_admin import firestore, credentials

# Use a service account.
cred = credentials.Certificate('credentials.json')

fb_app = firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello')
def hello():
    return 'world!'

@app.route('/getdocs')
def docs():
    index = db.collection("index")
    docs = index.stream()
    response = ""

    for doc in docs:
        response += (str(doc.id) + "<br>")
        for hits in doc.to_dict().values():
            for hit in hits:
                response += (hit + "<br>")
        response += "<br>"
    return response



if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
