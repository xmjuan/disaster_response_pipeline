import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import nltk
nltk.download(['wordnet','stopwords','punkt'])

import pickle

def load_data(database_filepath):
    """Method to create connection and load dataset from database
    
    Args:
    database_filepath: str
    
    Returns:
    X: pandas series
    y: pandas series
    df: dataframe
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM msg_df", con=engine)
    X = df['message']
    y = df.iloc[:, 4:]
    
    return X, y, df


def tokenize(text):
    """Method to tokenize and stem text
    
    Args:
    text: str
    
    Returns:
    text: str - tokenized
    """
    
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenization
    token_words = word_tokenize(text.lower())
    
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in token_words if not w in stop_words]
    
    # stemming and lemmatization
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    stemmed_text = []
    for word in filtered_text:
        stemmed = stemmer.stem(word)
        lemmatized = lemmatizer.lemmatize(stemmed).lower().strip()
        stemmed_text.append(lemmatized)
    
    return stemmed_text


def build_model():
    """Method to create random forest classifier and pipelines
    
    Args:
    None
    
    Returns:
    cv: GridSearch object
    """
    # create random forest classifier pipeline
    pipeline_rfc = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    # set parameters
    parameters = {'clf__estimator__n_estimators': [10, 20],
                  'clf__estimator__criterion': ['entropy']}
    
    # fit model
    cv = GridSearchCV(pipeline_rfc, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """Method to iterate classification report per target and overall accuracy
    
    Args:
    model: classification object
    X_test: panda series
    Y_test: panda series
    category_names: list
    
    Returns:
    None
    """
    
    y_pred = model.predict(X_test)
    # calculated overall accuracy
    print('Overall accuracy of current model is {:.3f}'.format((y_pred == Y_test).mean().mean()))
    
    # classification report per column
    col_names = Y_test.columns
    y_pred_df = pd.DataFrame(y_pred, columns=col_names)
    for col in y_pred_df.columns:
        results = classification_report(Y_test[col], y_pred_df[col])
        print('Feature:', col, '\n', results)


def save_model(model, model_filepath):
    """Method to save model to pickle file
    
    Args:
    model: random forest classifier object
    model_filepath: str
    
    Returns:
    None
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        evaluate_model(model, X_test, Y_test)

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