## Disaster_response_analyzer

# Analyzing message data for disaster response


# Summary of the project

The aim of this project is to build a model that based on the input message can precisely analyze what disaster category it belongs to.

We trained our model on message input and 36 response categories. We used  MultiOutputClassifier() for multi-target classification and RandomForestClassifier()  as a base estimator.

# Repo walkthrough
  - Data is located in `data` directory
  - Your model can be found in `models` directory. Please note, that due to size of the model you'll have to follow execution steps below in order to train your own model
  - App template and run entry point for the app are in `app`


# Execution steps

1. In order to train the model  you have to follow the next steps:
  - To run ETL pipeline that cleans data and stores in database:
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     
 - To run ML pipeline that trains classifier and saves:
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To launch the app, go to `app` directory and execute `run.py`


