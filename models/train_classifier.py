import sys
import joblib

import nltk
from sqlalchemy import create_engine, MetaData, Table
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    """

    :param database_filepath: path to dataFrame
    :return: column X and dateFrame Y, Names of the columns Y
    """
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    connection = engine.connect()
    
    metadata = MetaData()
    
    table = Table('InsertTableName', metadata, autoload=True, autoload_with=engine) 
    
    result = connection.execute(table.select()).fetchall()
    
    df = pd.DataFrame(result, columns=table.columns.keys())
    
    X = df['message']
    
    y = df.iloc[ : , 4: ]
    
    category_names=y.columns


    return X, y , category_names



def tokenize(text):
    """

    :param text: text to tokenize
    :return: cleaned text
    """
    # Define the regular expression pattern to remove non letters
    reg = r"[^a-zA-Z0-9]"
    
    # Replace non-letters  with spaces
    text = text.replace(reg,' ')
    
    # Convert the text to lowercase
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return  clean_tokens 


def build_model():
    """
    Building pipeline and doing gridsearch to build a model
    :return: model object
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        
        ('tfidf', TfidfTransformer()),
        
        ('moc', MultiOutputClassifier(RandomForestClassifier())) 
     ])
        

    param_grid = {
        
    'moc__estimator__n_estimators': [100],

    # you can another parametrs to increase accuracy
        
    # 'moc__estimator__max_depth': [None, 10],
        
    # 'moc__estimator__min_samples_split': [2, 5]
    
    }

    model = GridSearchCV(pipeline, param_grid=param_grid,verbose=3, cv=3)

    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
     Classification_report to evaluate the model
    :param model: model
    :param X_test: test values for x
    :param Y_test: test values for y
    :param category_names: name of columns in Y

    """
    
    pred_mat=model.predict(X_test)
    
    pred_df=pd.DataFrame(pred_mat,columns=category_names)
    
    for column in category_names: 
        
        class_report = classification_report(Y_test[column], pred_df[column], target_names=['0', '1'])
        
        print(class_report)
    



def save_model(model, model_filepath):
    """
    Saving model
    :param model: model
    :param model_filepath: path to save model

    """
    
    joblib.dump(model, model_filepath)


def main():
    """
    Launch all the functions, split train and test data for x and y, fit the model.
    """
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