import sys
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifier

import pickle

def load_data(database_filepath):
    """Load data using the .db file from the data processing step and get it ready for model building
    Args:
    database_filepath: The filepath of .db file from the data processing step
    
    Returns:
    X: data set of messages
    Y: data set of categories
    category_names: list of category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df.message.values
    Y = df[df.columns[4:]]
    category_names = list(Y.columns.values)
    return X, Y, category_names
    pass


def tokenize(text):
    """Tokenize message input
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass


def build_model():
    """Build model pipeline with grid search
    clf: choose an algorithm and use MultiOutputClassifier for multiclass model
    pipeline: combine clf with other text processing steps in one pipeline
    parameters: choose the parameters and values for grid search
    cv: finalize model pipeline with grid search
    """
    clf = MultiOutputClassifier(RidgeClassifier())
    pipeline_best = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', clf)
      ])
    parameters = {
        'vect__min_df': [1, 2],
        'tfidf__smooth_idf': [True, False],
        'clf__estimator__random_state':[True, False],
        'clf__estimator__alpha':[1.0, 2.0]
    }

    cv = GridSearchCV(pipeline_best, param_grid=parameters)
    model = cv
    return model
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the precision, recall and f1 score for the test set
    Args:
    model: the model fit by the train set
    X_test: test set of X, used to predict Y values -- Y_pred
    Y_test: test set of Y (true Y values), used to compare with Y_pred (predicted Y values) in the classification           report
    category_names: a collection of all category names, being used in the classification_report output so there will be     evaluation of each category
    
    Returns:
    A print-out of the classification_report, including precision, recall and f1 score for the test set from each category
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))
    pass


def save_model(model, model_filepath):
    """Save model as a pickle file
    Args:
    model: the model fit by the train set
    model_filepath: the filepath of model
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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