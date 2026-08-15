# Cyberbullying Detection System

A web-based cyberbullying detection system developed as an MCA academic project.

The system analyzes text messages using a combination of a machine-learning classifier and rule-based text analysis. The hybrid approach is designed to identify potentially harmful or cyberbullying-related messages while avoiding the assumption that every offensive word automatically represents cyberbullying.

---

## Project Overview

Cyberbullying can involve direct insults, abusive language, harassment, and other forms of harmful communication.

This project provides a text-based detection system that combines:

- Machine Learning classification
- TF-IDF text representation
- Toxic-word detection
- Targeted-abuse detection
- Hybrid decision-making
- SQLite-based incident logging
- A web-based frontend

The system returns one of three main outcomes:

- **Safe**
- **Cyberbullying**
- **Review**

`Review` is used when offensive language is detected but the available rules do not clearly establish targeted cyberbullying.

---

## System Architecture

```text
                    User
                     │
                     ▼
             Web Frontend
          HTML / CSS / JavaScript
                     │
                     ▼
                Flask API
                 app.py
                     │
                     ▼
             Text Preprocessing
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Machine Learning       Rule-Based Analysis
      Prediction                  │
          │              ┌────────┼────────┐
          │              │        │        │
          │              ▼        ▼        ▼
          │          Toxic Words  Targeted Abuse
          │
          └──────────┬───────────┘
                     ▼
             Hybrid Decision
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Safe     Review   Cyberbullying
                                │
                                ▼
                         SQLite Database

Main Features
1. Machine Learning Detection

The application uses a trained machine-learning classifier to classify processed text.

The Flask application currently loads:

models/SGDClassifier_model.pkl

The classifier produces a binary prediction:

0 → Safe
1 → Cyberbullying

The ML model is used together with the rule-based detection layer to produce the final result.

2. TF-IDF Text Representation

The input text is converted into numerical features using a trained TF-IDF vectorizer.

The application loads:

models/vectorizer.pkl

The vectorizer transforms incoming text before it is passed to the machine-learning classifier.

The application uses transform() on new messages rather than fitting a new vectorizer for every individual message.

3. Text Preprocessing

Text preprocessing is implemented in:

hybrid_engine.py

The preprocessing stage includes operations such as:

Removing user mentions
Removing unwanted characters
Converting text to lowercase
Tokenization
Lemmatization

The processed text is then supplied to the trained TF-IDF vectorizer.

4. Toxic Word Detection

The project contains a rule-based toxic-word detector.

The word list is stored in:

bad_words.txt

The system loads the words from this file and checks incoming messages using token-aware matching.

This rule-based layer provides an additional detection mechanism alongside the machine-learning classifier.

Important

The presence of an offensive word does not automatically mean that the final result is cyberbullying.

The system also considers whether potentially abusive language appears to be targeted.

5. Targeted Abuse Detection

The project contains a targeted-abuse detection function in:

hybrid_engine.py

It looks for patterns indicating that potentially abusive language is directed toward another person.

Examples of patterns handled by the current rule-based system include expressions involving:

you are ...
you look ...
your ...
nobody likes you
go die
kill yourself
shut up

The system does not treat the presence of the word you alone as sufficient evidence of cyberbullying.

Hybrid Decision System

The final decision is produced by:

hybrid_decision()

This function combines:

Machine-learning prediction
Toxic-word detection
Targeted-abuse detection
Case 1 — Machine Learning detects cyberbullying

If:

ML prediction = 1

the final classification is:

Cyberbullying
Case 2 — ML predicts safe but targeted toxic language is detected

If:

ML prediction = 0
Toxic word detected = Yes
Targeted abuse = Yes

the hybrid system classifies the message as:

Cyberbullying

This allows the rule-based layer to identify cases that the ML classifier may miss.

Case 3 — Toxic language without clear targeting

If:

ML prediction = 0
Toxic word detected = Yes
Targeted abuse = No

the system returns:

Review

This is important because offensive language does not always constitute cyberbullying.

For example, an offensive word used in a non-targeted context should not automatically be treated as a personal cyberbullying incident.

Case 4 — No detected harmful indicators

If:

ML prediction = 0
Toxic word detected = No

the result is:

Safe
API

The Flask application provides the following main endpoint:

POST /validate-comment

The endpoint accepts a JSON request.

Example Request
{
    "text": "Have a great day!"
}
Example Safe Response
{
    "prediction": 0,
    "label": "safe",
    "toxic_word_detected": false,
    "targeted_abuse": false,
    "status": "allowed",
    "message": "No cyberbullying detected."
}
Example Cyberbullying Response
{
    "prediction": 1,
    "label": "cyberbullying",
    "toxic_word_detected": true,
    "targeted_abuse": true,
    "status": "blocked",
    "message": "Cyberbullying detected."
}

The exact message returned can depend on which part of the hybrid detection system produced the final decision.

Web Frontend

The project includes a browser-based frontend.

The frontend is located in:

templates/
static/
HTML
templates/index.html
templates/result.html
CSS
static/css/style.css
JavaScript
static/js/script.js

The frontend communicates with the Flask backend and displays the analysis result to the user.

Database

The project uses SQLite for storing detected cyberbullying reports.

Database file:

cyberbullying.db

The database contains a reports table.

Reports Table
Column	Type	Purpose
id	INTEGER	Unique report ID
content	TEXT	Detected message
user_ip	TEXT	IP address associated with the request
account_name	TEXT	Account/user name
timestamp	DATETIME	Time of the report

When the final hybrid decision is classified as cyberbullying, the message is recorded in the database.

Project Structure
CYBERBULLYING-PROJECT/
│
├── app.py
├── cyberbullying-project.py
├── hybrid_engine.py
├── train_model.py
│
├── bad_words.txt
├── cyberbullying.db
├── tfidfvectorizer11.pkl
│
├── data1.csv
├── model.ipynb
├── stopwords.txt
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── LinearSVC_model.pkl
│   ├── SGDClassifier_model.pkl
│   └── vectorizer.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── templates/
    ├── index.html
    └── result.html
Important Files
app.py

Main Flask application.

Responsibilities include:

Starting the Flask server
Loading the trained model
Loading the TF-IDF vectorizer
Receiving text from the frontend/API
Performing prediction
Calling the hybrid decision engine
Logging cyberbullying reports
hybrid_engine.py

Contains the hybrid text-analysis logic.

Responsibilities include:

Text preprocessing
Toxic-word detection
Targeted-abuse detection
Combining ML and rule-based results
train_model.py

Contains the model-training process used to generate the trained machine-learning resources.

model.ipynb

Jupyter Notebook containing the original machine-learning experimentation/training work.

data1.csv

Dataset used for machine-learning training.

bad_words.txt

Contains the rule-based list of explicit toxic words.

cyberbullying.db

SQLite database used to store detected cyberbullying reports.

models/

Contains the trained machine-learning resources.

The Flask application currently uses:

SGDClassifier_model.pkl
vectorizer.pkl

LinearSVC_model.pkl is also present in the project as a trained model resource.

Installation
1. Clone the Repository
git clone https://github.com/Pkrithik/CYBERBULLYING-PROJECT.git
2. Enter the Project Directory
cd CYBERBULLYING-PROJECT
3. Install Dependencies

If requirements.txt contains the required packages:

pip install -r requirements.txt

Alternatively, the main packages used by the project include:

pip install flask flask-cors scikit-learn nltk pandas numpy

Additional packages may be required for the external API functionality contained in cyberbullying-project.py.

Running the Application

From the project directory:

python app.py

The Flask development server should start at:

http://127.0.0.1:5000

Open the address in a web browser.

Testing the API

PowerShell example:

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
Machine Learning Resources

The project contains the following trained resources:

models/SGDClassifier_model.pkl
models/LinearSVC_model.pkl
models/vectorizer.pkl

The current Flask application loads:

SGDClassifier_model.pkl
vectorizer.pkl

The vectorizer is used to transform incoming text into the feature representation expected by the classifier.

External API Functionality

The original project also contains external-platform integration code in:

cyberbullying-project.py

This code contains functionality related to:

YouTube comments
Instagram comments
IP address retrieval

These functions are separate from the primary local text-validation flow provided by:

app.py

External API credentials should be supplied through environment variables rather than being stored directly in source code.

Limitations

This is an academic machine-learning project and the detection system is not guaranteed to identify every instance of cyberbullying.

Possible limitations include:

False positives
False negatives
Context-dependent language
Sarcasm
Slang
Misspellings
New or previously unseen abusive expressions
Limitations of the training dataset
Limitations of rule-based pattern matching

The hybrid approach is intended to improve detection by combining machine learning with explicit rule-based checks.

Privacy and Security Considerations

The application stores detected cyberbullying reports in a local SQLite database.

The database may contain:

Message content
IP address
Account name
Timestamp

Therefore, the database should be handled carefully when deploying the application in a real-world environment.

The project should not be considered production-ready without additional security, privacy, authentication, access-control, and data-protection measures.

Future Enhancements

Possible future improvements include:

Improved contextual cyberbullying detection
Larger and more diverse training datasets
Better handling of slang and misspellings
Multilingual detection
Improved sarcasm detection
Advanced NLP models
Visualization of detection statistics
Administrative reporting interface
Improved social-media integration
More sophisticated harassment and abuse classification
Academic Project

Project Title: Cyberbullying Detection System

Degree: Master of Computer Applications (MCA)

Domain: Machine Learning / Natural Language Processing / Web Application

Purpose: Academic Project