import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    column_sum = df.iloc[:, 4:].sum()
    df_1 = pd.DataFrame(column_sum, columns=['freq']).reset_index().sort_values(by='freq', ascending=False)

    df_1.columns = ['Categories', 'Value']
    genre_counts = df_1['Categories'].head(4)
    genre_names = df_1['Value'].head(4)
    counts = df_1['Categories'].tail(4)
    names = df_1['Value'].tail(4)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    y=genre_names,
                    x=genre_counts,
                )
            ],

            'layout': {
                'title': 'Most frequent Categories ',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    y=names,
                    x=counts,
                )
            ],

            'layout': {
                'title': 'Least frequent Categories ',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
