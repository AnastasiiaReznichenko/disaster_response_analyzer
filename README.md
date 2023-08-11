## Disaster_response_analyzer

# Analyzing message data for disaster response


# Summary of the project

The project's core mission is to create an advanced model that accurately categorizes incoming messages into specific disaster types. By achieving this, we can rapidly connect each message to the right relief agency, enabling quicker and more precise disaster response efforts. This time-saving process is vital in emergency situations where swift action can make a life-saving difference.

We trained our model on message input and 36 response categories. We used  MultiOutputClassifier() for multi-target classification and RandomForestClassifier()  as a base estimator.

# Repo walkthrough
  - Data is located in `data` directory:
      1. `disaster_messages.csv` is the raw file that contains plain text messages that we will be trying to categorize during model training
      2. `disaster_categories.csv` has a list of categories that should be mapped to messages to complete training dataset  
  - Your model can be found in `models` directory. Please note, that due to the size of the model, you'll have to follow the execution steps below in order to train your own model
  - App template and run entry point for the app are in `app`


# Execution steps

1. In order to train the model  you have to follow the next steps:
  - To run ETL pipeline that cleans data and stores in the database:
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     
 - To run ML pipeline that trains classifier and saves:
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. To launch the app, go to `app` directory and execute `run.py`


