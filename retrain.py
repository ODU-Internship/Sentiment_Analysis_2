import pymongo
import pandas as pd
#UTC time
import time
from train_model import retrain, replace_model
from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
from json import dumps

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
    if(col_no<=10000 or checkIfMidnight()): #TODO correct logic
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



app = Flask(__name__)
api = Api(app)



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


api.add_resource(Messages_retrain, '/retrain')
api.add_resource(Messages_replace, '/replace')

if __name__ == '__main__':
    app.run(debug=True)
