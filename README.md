# Disaster-Response-Pipeline-Project
## Project Overview
This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

## Project Components:
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

## Data:
- messages.csv
- categories.csv


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
 `

3. Go to http://0.0.0.0:3001/
