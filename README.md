# CYBERBULLYING-PROJECT
COLLEGE PROJECT
# CyberShield – Cyberbullying Detection System

CyberShield is an MCA project for detecting potentially harmful and cyberbullying-related messages using a combination of **Machine Learning and rule-based text analysis**.

The system provides a web interface where a user can enter a message and receive an analysis of the message.

---

## Project Overview

The system uses a hybrid detection approach instead of relying only on a machine-learning classifier.

The message passes through multiple stages:

1. Text preprocessing
2. Machine Learning classification
3. Explicit toxic-word detection
4. Targeted-abuse detection
5. Hybrid decision-making
6. Cyberbullying report storage in SQLite

The final result can be classified as:

- **Cyberbullying**
- **Safe**
- **Review**

The `Review` category is used when offensive language is detected but the system cannot confidently determine that the message is targeted cyberbullying.

---

## Main Technologies

### Backend

- Python
- Flask
- Flask-CORS
- Scikit-learn
- NLTK
- SQLite

### Frontend

- HTML
- CSS
- JavaScript

### Machine Learning

- TF-IDF Vectorization
- SGDClassifier
- Scikit-learn

### Database

- SQLite

---

## Project Structure

```text
CYBERBULLYING-PROJECT/
│
├── app.py
├── cyberbullying-project.py
├── hybrid_engine.py
├── train_model.py
├── bad_words.txt
├── cyberbullying.db
├── tfidfvectorizer11.pkl
│
├── models/
│   ├── SGDClassifier_model.pkl
│   ├── LinearSVC_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       └── script.js
│
├── data1.csv
├── model.ipynb
├── stopwords.txt
└── .gitignore


How the System Works
1. User Input

The user enters a message through the CyberShield web interface.

Example:

you are ugly

The frontend sends the message to the Flask backend.

2. Flask API

The backend is implemented in:

app.py

The main API endpoint is:

POST /validate-comment

The endpoint accepts JSON data:

{
    "text": "you are ugly"
}
3. Text Preprocessing

The message is processed by:

hybrid_engine.py

The preprocessing stage performs operations such as:

Removing mentions
Removing unwanted characters
Converting text to lowercase
Tokenization
Lemmatization

The same preprocessing approach is intended to be used before ML prediction.

4. Machine Learning Detection

The processed message is converted into numerical features using the trained TF-IDF vectorizer.

The vectorizer is loaded from:

models/vectorizer.pkl

The trained classification model is loaded from:

models/SGDClassifier_model.pkl

The ML classifier produces a binary prediction:

0 → Safe
1 → Cyberbullying
5. Rule-Based Detection

The project also contains rule-based detection in:

hybrid_engine.py
Toxic Word Detection

The system loads explicit words from:

bad_words.txt

The words are checked using token-aware matching.

This allows the system to identify explicit toxic language that the ML model may not detect correctly.

Targeted Abuse Detection

The system also checks whether potentially abusive language is directed toward another person.

Examples of targeted patterns include phrases such as:

you are ...
you look ...
your ...
nobody likes you
go die
kill yourself
shut up

Targeted detection is not based on the word you alone.

The purpose is to determine whether potentially offensive language is actually being directed at a person.

6. Hybrid Decision Engine

The final decision is made by:

hybrid_decision()

The system combines:

ML prediction
Toxic-word detection
Targeted-abuse detection

The basic decision logic is:

ML detects cyberbullying
ML = 1

Final result:

Cyberbullying
ML is safe but toxic language is targeted
ML = 0
Toxic words = Yes
Targeted abuse = Yes

Final result:

Cyberbullying
Toxic language exists but targeting is unclear
ML = 0
Toxic words = Yes
Targeted abuse = No

Final result:

Review

This avoids automatically classifying every offensive word as cyberbullying.

No harmful indicators
ML = 0
Toxic words = No

Final result:

Safe
Classification Results

The API can return information such as:

{
    "prediction": 1,
    "label": "cyberbullying",
    "toxic_word_detected": true,
    "targeted_abuse": true,
    "status": "blocked",
    "message": "Cyberbullying detected."
}

For a safe message, the result can contain:

{
    "prediction": 0,
    "label": "safe",
    "toxic_word_detected": false,
    "targeted_abuse": false,
    "status": "allowed",
    "message": "No cyberbullying detected."
}
Database

The project uses:

cyberbullying.db

SQLite is used to store detected cyberbullying incidents.

The main table is:

reports

The table contains:

Column	Type	Description
id	INTEGER	Unique report ID
content	TEXT	Detected message
user_ip	TEXT	IP address
account_name	TEXT	Account/user name
timestamp	DATETIME	Time of detection

When the final decision is cyberbullying, the message can be stored in the database.

Web Interface

The frontend is served by Flask.

The main page is:

templates/index.html

The result page is:

templates/result.html

Frontend styling is located at:

static/css/style.css

Frontend JavaScript is located at:

static/js/script.js

The interface allows the user to:

Enter a message
Submit the message
Send the message to the Flask API
Receive the detection result
View the classification details
Running the Project
Step 1 – Install Python

Make sure Python is installed.

Check:

python --version
Step 2 – Install Required Packages

Run:

pip install flask flask-cors scikit-learn nltk pandas numpy

If required by the existing project:

pip install google-api-python-client requests
Step 3 – Start the Flask Application

From the project directory:

python app.py

The server should start at:

http://127.0.0.1:5000

Open this address in a browser.

Testing the API

PowerShell can be used to test the API.

Example:

Invoke-RestMethod `
-Uri "http://127.0.0.1:5000/validate-comment" `
-Method POST `
-ContentType "application/json" `
-Body '{"text":"Have a great day!"}'

Another example:

Invoke-RestMethod `
-Uri "http://127.0.0.1:5000/validate-comment" `
-Method POST `
-ContentType "application/json" `
-Body '{"text":"you are ugly"}'
Machine Learning Model Files

The trained model files are stored in:

models/

Currently included model resources include:

SGDClassifier_model.pkl
LinearSVC_model.pkl
vectorizer.pkl

The Flask application currently loads:

SGDClassifier_model.pkl
vectorizer.pkl
Training

The project contains:

train_model.py

and the original notebook:

model.ipynb

The training dataset is:

data1.csv

The purpose of the training script is to generate the machine-learning model and vectorizer used by the application.

External API Integration

The original project also contains integration functionality for external platforms.

These include:

YouTube Data API
Instagram Graph API
IP address lookup through IPify

The external-platform functionality is primarily contained in:

cyberbullying-project.py

The Flask web interface currently uses the local message-validation API provided by:

app.py
Security

API credentials should not be hardcoded into the source code.

Environment variables should be used for external API credentials.

For example:

YOUTUBE_API_KEY

The .gitignore file should also be used to prevent sensitive configuration files and unnecessary generated files from being committed.

Important Limitations

CyberShield is an academic MCA project and its classification results should not be considered perfect.

Machine-learning predictions can produce:

False positives
False negatives

Rule-based detection can also have limitations because offensive language can depend heavily on context.

Therefore, the system uses a hybrid approach to improve detection rather than relying on a single technique.

Project Objective

The primary objective of this project is to develop a system capable of identifying potentially cyberbullying messages by combining:

Machine Learning
        +
TF-IDF Text Representation
        +
Rule-Based Toxic Word Detection
        +
Targeted Abuse Detection
        =
Hybrid Cyberbullying Detection
Academic Information

Project: Cyberbullying Detection System

Degree: Master of Computer Applications (MCA)

Purpose: Academic Mini Project / Project Work

Domain: Machine Learning / Natural Language Processing / Web Application