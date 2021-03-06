import re

import pandas as pd

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb

import warnings
import joblib

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO

from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

model = joblib.load('finalized_model.model')


@app.route('/')
def home():
    return render_template('index.html')


@socketio.on('/predict')
def predict():
    """
    For rendering results on HTML GUI
    """
    stop_words = stopwords.words('english')
    wnl = WordNetLemmatizer()

    tweet = request.get_json()

    comment = tweet['text']
    print(comment)

    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(comment).lower()).strip()
    text = [wnl.lemmatize(i) for i in text.split(' ') if i not in stop_words]
    text = ' '.join(text)
    text = [text]

    train = pd.read_csv('train.csv')
    train = train[['comment_text', 'toxic']]

    cv = CountVectorizer(binary=True)
    cv.fit(train['comment_text'])

    X = cv.transform(text)

    prediction = model.predict(X)

    if prediction == 1:
        output = 'Toxic'
        print(output)
    else:
        output = 'Positive'
        print(output)

    send(output)


if __name__ == "__main__":
    socketio.run(app, debug=True)
