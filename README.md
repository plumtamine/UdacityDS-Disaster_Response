# UdacityDS-Disaster_Response
Disaster Response Pipelines - Creating data ETL pipeline, Machine Learning pipeline and publishing web app to query for model prediction.

# Prerequisites
Python ver 3.x.

# Project Understanding
This Disaster Response project is aiming to analyze message texts, predict disaster categories based on messages to facilitate faster response and provide a web app for message-category lookup. There are 3 modules to be done to achieve this goal: 
1. ETL pipeline: The raw data is not in a perfect form to consume directly. We create a pipeline to clean, manipulate and store data, as data preparation for modeling.
2. Machine Learning pipeline: When creating model for data, we run a large amount of iterations to find the best parameters of model. Instead of repeating similar chunk of codes every time, we simply the process with Machine Learning pipeline. In addition, this helps to process train data in each fold to avoid data leak.
3. Web Deployment: With both the data and model ready from the back end, we use plotly.js and flask to deploy the dashboard and search bar to query messages.

# ETL pipeline
The ETL script, process_data.py, takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.
* The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. GridSearchCV is used to find the best parameters for the model.


# Machine Learning pipeline
The machine learning script, train_classifier.py, takes the database file path and model file path, creates and trains a classifier, and stores the classifier into a pickle file to the specified model file path.
* The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text.
* The script builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the dataset. GridSearchCV is used to find the best parameters for the model.
* The TF-IDF pipeline is only trained with the training data. The f1 score, precision and recall for the test set is outputted for each category.

# Web Deployment
The web app, run.py, initiates the webpage. The main page includes two visualizations using data from the SQLite database.
* When a user inputs a message into the app, the app returns classification results for all 36 categories.
