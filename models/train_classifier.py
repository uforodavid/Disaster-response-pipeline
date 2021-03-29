import sys
# import libraries
import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import pickle
import warnings
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, precision_score
import os
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import nltk
from nltk.corpus import stopwords

    #load_data()
def load_data(database_filepath):
    engine = create_engine('sqlite:////Users/davideffiong/Documents/Disaster-Response-Pipeline/DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM disaster_table", engine)
    X = df['message'] #feature variable
    Y = df.iloc[:,4:] #target variable
    return X, Y

def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():
    #split data into training and test data set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
    #Random Forest Classifier pipeline

    pipeline_rfc = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',  MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters_rfc = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 20]
    }

    cv_rfc = GridSearchCV(pipeline_rfc, param_grid = parameters_rfc)
    return cv_rfc

    
def plot_scores(Y_test, Y_pred):
    i = 0
    for col in Y_test:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i + 1
    accuracy = (Y_pred == Y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))

    
def evaluate_model(model, X_test, Y_test, category_names):
    # Get results and add them to a dataframe.
    # Predicting using the first tuned model 
    Y_pred = cv_rfc.predict(X_test)
    plot_scores(Y_test, Y_pred)


def save_model(model, model_filepath):
    # Create a pickle file for the model
    file_name = 'classifier.pkl'
    with open (file_name, 'wb') as f:
        pickle.dump(cv_ada, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()