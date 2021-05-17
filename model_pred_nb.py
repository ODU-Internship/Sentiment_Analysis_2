import pandas as pd
import pymongo
#UTC time
import time
from train_model import retrain, replace_model
from preprocess import convert_lower, remove_special
import re
import numpy as np

from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from json import dumps



import pickle
filename = 'nb_model.sav' #TODO change to models/model1.sav

loaded_model = pickle.load(open(filename, 'rb'))

"""
SAMPLE EMAIL INPUT
text = {'emails':["invitation strategy workshop february manchester workshop forward colleague results campaign workshop reminder content workshop reminder ca cs dad ca trouble viewing view results campaign workshop reminder content workshop reminder cs dad ca assets images azure cs dad ca azure join cs dad ca manchester discuss journey unleash benefits powered by azure workshop control optimise strategy tickets cs dad ca february discussing best working opportunities digital give tools how embrace perfect environment wednesday february where sir busby way manchester further information found control optimise strategy tickets cs dad ca please note places join workshop manchester february control optimise strategy tickets cs dad ca make february workshop interest march where street campus birmingham join workshop birmingham control optimise strategy tickets cs dad ca assets images footer cs dad ca action campaign workshop reminder content workshop reminder ca cs dad ca road bb sr","hi guys please details hosted building hyper infrastructure location mandatory since client accept his other than machine operating newer purposes cpu disk worry about partitioning handle needs accessible networks also needs ports user root rights client machine operating newer purposes cpu disk needs accessible networks especially user rights client thank engineer","hello please be kind source quotation macedonian suppliers for equipment destined fit out one large meeting room equipment should be one shopping list or equivalent projector audio mixer mixer channels audio input chime music only output each input controlled separately volume override option receiver wireless microphone receiver mw wireless microphone mw tuner sd tuner source loudspeaker speaker case connections projects lead", "contract status issue hi we have contract snow we have marked terminated yesterday spite we have received trigger today when checked contract had changed status expired we have again marked terminated today you know why happened can we prevent happening same issue might apply other contracts we know thank you for looking into kind regards administrator content uploads take step today think before you print"]}
"""

myclient = pymongo.MongoClient("mongodb+srv://alpha:root@cluster0.buscd.mongodb.net/alpha?retryWrites=true&w=majority")
mydb = myclient["alpha"]

target = 'label'
feature= 'message'

def checkIfMidnight():
    return (time.time() % 86400 >= 0 or time.time() % 86400 <= 3)

def retrain_validated():
    mycol = mydb ["retrains"]
    cursor = mycol.find()
    list_cur = list(cursor)
    df = pd.DataFrame(list_cur)
    df = df[['message','label']]
    df['label'] = [i[0] for i in df['label']]
    col_no = mycol.count_documents({})
    print("The number of documents in collection : ",col_no)
    if(col_no>=10000 or checkIfMidnight()): #TODO correct logic
        #refit()
        print("retraining with validated data")
        return df
    else:
        return pd.DataFrame()

def retrain_uploaded():
    mycol = mydb ["trains"]
    cursor = mycol.find()
    list_cur = list(cursor)
    df = pd.DataFrame(list_cur)
    df = df[['message','label']]
    df['label'] = [i[0] for i in df['label']]
    col_no = mycol.count_documents({})
    if(col_no >1):
        print("retraining with uploaded data")
        return df
    else:
        return pd.DataFrame()


df = pd.DataFrame()
col = 'emails'
embed_dim = 128
lstm_out = 196
max_fatures = 2000


def process_emails(text_list,col=col,df=df):
    df['emails'] = text_list
    df_sub = convert_lower(df, col)
    df_sub = remove_special(df_sub, col)
    return df_sub


def get_predictions(X,loaded_model=loaded_model):
    predictions = loaded_model.predict_proba(X)
    #print(predictions)
    probs = np.max(predictions, axis=1)*100
    probs =[str(round(i,2)) for i in probs]
    #print(probs)
    preds = np.argmax(predictions, axis=1)
    #print(preds)
    sentiment_map = {1:'Positive', 0:'Negative - needs attention'}
    senti_preds = [sentiment_map[i] for i in preds]
    #print(senti_preds)
    result = {'probability':probs, 'sentiment':senti_preds}
    #print(result)
    return result


app = Flask(__name__)
api = Api(app)



class Messages(Resource):
    def get(self):
        return "get request not supported"

    def put(self):
        #text_list= request.get_json()['emails']
        text_list= request.json
        text_list = text_list['emails']
        #print(text_list)
        X = process_emails(text_list)
        result = get_predictions(list(X['emails']))
        return result


class Messages_retrain(Resource):
    def get(self):
        return "get request not supported"

    def put(self):

        df2 = retrain_uploaded()
        print(df2)
        if not (df2.empty):
            q2 = retrain(df2,target,feature)
            #print(q2)
            return q2

        df1 = retrain_validated()
        print(df1)
        if not (df1.empty):
            q = retrain(df1,target, feature)
            return q
            #print(q)

        else:
            return "Upload file to use for retraining"


class Messages_replace(Resource):
    def get(self):
        return "get request not supported"

    def put(self):

        merge_model= request.json
        merge_model_ans = merge_model['merge']
        if(merge_model_ans == 'yes'):
            v = replace_model()
            return v
        else:
            return "model not replaced"



api.add_resource(Messages, '/predict')
api.add_resource(Messages_retrain, '/retrain')
api.add_resource(Messages_replace, '/replace')

if __name__ == '__main__':
    app.run(debug=True)
