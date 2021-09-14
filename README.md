# disaster_response_pipeline

### Project Introduction:
This is the second project in Udacity Data Scientist Nanodegree Program.

### Installation and Library:
python==3.7.10 <br>
pandas==0.23.3 <br>
Flask==0.12.5 <br>
numpy==1.12.1 <br>
scikit-learn==0.19.1 <br>
SQLAlchemy==1.2.19 <br>
snowballstemmer==2.1.0 <br>
nltk==3.2.5 <br> 

### Motivation:
This disaster response project is classifying disaster message by:
- creating ETL pipeline (collect, clean and store data in database)
- creating Machine Learning pipeline
- building a web app

### File description:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

### Key process:

- To run ETL pipeline that cleans data and stores in database
    	`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run web app.
    `python run.py`

- Go to http://0.0.0.0:3001/

Author:
Ximin Juan

Acknowledgements:
Udacity Data Scientist Program