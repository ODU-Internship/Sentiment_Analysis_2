from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

from preprocess import convert_lower, remove_special

import tensorflow as tf
import pandas as pd
import re
import numpy as np

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps


"""
SAMPLE EMAIL INPUT
text = {'emails':["invitation strategy workshop february manchester workshop forward colleague results campaign workshop reminder content workshop reminder ca cs dad ca trouble viewing view results campaign workshop reminder content workshop reminder cs dad ca assets images azure cs dad ca azure join cs dad ca manchester discuss journey unleash benefits powered by azure workshop control optimise strategy tickets cs dad ca february discussing best working opportunities digital give tools how embrace perfect environment wednesday february where sir busby way manchester further information found control optimise strategy tickets cs dad ca please note places join workshop manchester february control optimise strategy tickets cs dad ca make february workshop interest march where street campus birmingham join workshop birmingham control optimise strategy tickets cs dad ca assets images footer cs dad ca action campaign workshop reminder content workshop reminder ca cs dad ca road bb sr","hi guys please details hosted building hyper infrastructure location mandatory since client accept his other than machine operating newer purposes cpu disk worry about partitioning handle needs accessible networks also needs ports user root rights client machine operating newer purposes cpu disk needs accessible networks especially user rights client thank engineer","hello please be kind source quotation macedonian suppliers for equipment destined fit out one large meeting room equipment should be one shopping list or equivalent projector audio mixer mixer channels audio input chime music only output each input controlled separately volume override option receiver wireless microphone receiver mw wireless microphone mw tuner sd tuner source loudspeaker speaker case connections projects lead", "contract status issue hi we have contract snow we have marked terminated yesterday spite we have received trigger today when checked contract had changed status expired we have again marked terminated today you know why happened can we prevent happening same issue might apply other contracts we know thank you for looking into kind regards administrator content uploads take step today think before you print"]}
"""


df = pd.DataFrame()
col = 'emails'
embed_dim = 128
lstm_out = 196
max_fatures = 2000


def process_emails(text_list, col=col, df=df):
    df['emails'] = text_list
    df_sub = convert_lower(df, col)
    df_sub = remove_special(df, col)
    # remove stop words -
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(df[col].values)
    X = tokenizer.texts_to_sequences(df[col].values)
    X = pad_sequences(X)
    return X


def make_model(embed_dim=embed_dim, lstm_out=lstm_out):
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=128))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


model2 = make_model()
checkpoint_filepath = 'trainedWeights/'
model2.load_weights(checkpoint_filepath)


def get_predictions(X, model2=model2):
    predictions = model2.predict(X, batch_size=X.shape[0])
    # print(predictions)
    preds = np.array(predictions)
    probs = np.max(preds, axis=1)*100
    probs = [str(round(i, 2)) for i in probs]
    # print(probs)
    preds = np.argmax(preds, axis=1)
    # print(preds)
    sentiment_map = {1: 'Positive', 0: 'Negative - needs attention'}
    senti_preds = [sentiment_map[i] for i in preds]
    # print(senti_preds)
    result = {'probability': probs, 'sentiment': senti_preds}
    # print(result)
    return result


app = Flask(__name__)
api = Api(app)


class Messages(Resource):
    def get(self):
        return "get request not supported"

    def put(self):
        #text_list= request.get_json()['emails']
        text_list = request.json
        text_list = text_list['emails']
        # print(text_list)
        X = process_emails(text_list)
        result = get_predictions(X)
        return result


api.add_resource(Messages, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
