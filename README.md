# Disaster Response Pipeline Project
## Problem Statement:

The objective is to develop a web application that can classify emergency messages into various categories and display relevant visualizations. The project involves several components:

ETL Pipeline - This component will clean, transform, and store the data into a database.
ML Pipeline - This involves training a machine learning model to classify messages based on the provided categories.
Flask Web App - A web application that allows users to input new messages and get classification results, along with visualizations of the data.
Key tasks include:

Building and testing an ETL pipeline to prepare the data.
Developing a machine learning pipeline to train a classifier and evaluate its performance.
Creating a Flask web app to deploy the model and provide a user interface for message classification.


## Dataset Overview:

The Disaster Response Pipeline project utilizes two key datasets: messages.csv and categories.csv. The messages.csv dataset includes fields such as id, message, original, and genre, while the categories.csv dataset contains id and categories. The goal is to merge these datasets to create a comprehensive dataset for training and evaluating a machine learning model. The data will be used to build a pipeline that processes emergency messages and classifies them into multiple categories.


## Results:
Upon completion of the project, I will have a functional web application capable of classifying emergency messages into multiple categories. The ETL pipeline will have cleaned and prepared the data, while the ML pipeline will have trained and evaluated a classifier. The web app will allow users to input messages, receive classification results, and view data visualizations. Additionally, the project will emphasize code quality and effective data processing, with no minimum performance metrics required but with a focus on optimizing model accuracy, precision, and recall.

## Project Components:
1. ETL Pipeline:
- ETL Pipeline Preparation.ipynb

2. ML Pipeline:
- ML Pipeline Preparation.ipynb

3. Flask Web App:
- Process_data.py
- Train_classifier.py
- Run.py

## Data:
1. messages.csv
* id	
* message	
* original	
* genre
2. categories.csv
* id
* categories


