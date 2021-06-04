import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer

from preprocess import convert_lower, remove_special
import numpy as np
from sklearn import metrics
import pickle
import os

if(os.path.exists("models/model2.sav")):
    filename = 'models/model2.sav'
else:
    filename = 'models/model1.sav'
new_filename = 'models/model2.sav'


def replace_model():
    if os.path.exists('models/model2.sav') and os.path.exists('models/model1.sav'):
        os.remove("models/model1.sav")
        os.rename('models/model2.sav','models/model1.sav')
        #os.remove("models/model2.sav")
        return "Model merge successful"

    if not os.path.exists('models/model1.sav'):
        os.rename('models/model2.sav','models/model1.sav')
        return "Renamed model"

    if not os.path.exists("models/model2.sav"):
        return "No model to merge"


def retrain(df,target,feature):
    df = convert_lower(df, feature)
    df= remove_special(df, feature)

    """
    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english',lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())])
    """
    X = df[feature]
    y = df[target]
    label_dict = {'Good':1, 'Bad':0}
    y_label = [label_dict[i] for i in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.33, random_state=42)

    #count_vect = CountVectorizer()
    #X_train_counts = count_vect.fit_transform(X_train)
    #tf_transformer = pickle.load(open("models/tfidf.pickle", 'rb'))#TfidfTransformer(use_idf=False).fit(X_train_counts)
    test_vec = pickle.load(open("models/hash.pickle", 'rb'))
    X_train_vec = test_vec.transform(X_train)
    #clf = MultinomialNB().fit(X_train_tfidf,  y_train)

    loaded_model = pickle.load(open(filename, 'rb'))

    loaded_model.partial_fit(X_train_vec,  y_train)

    #X_new_counts = count_vect.transform(X_test)
    X_new_vec = test_vec.transform(X_test)
    predicted = loaded_model.predict(X_new_vec)

    acc = np.mean(predicted == y_test)
    print('Retrain successful')
    pickle.dump(loaded_model, open(new_filename, 'wb'))
    return(f"Accuracy: {acc}")#,metrics.classification_report(y_test, predicted,target_names=[str(i) for i in list(y.unique())]))
