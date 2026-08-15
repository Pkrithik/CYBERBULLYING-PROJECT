import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import pickle
import os

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Centralized text cleaning used for both training and inference.
    Fixes the old bug that deleted words <= 3 characters.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove @mentions
    text = re.sub(r"@[\w]*", "", text)
    # Remove non-alphabet characters (except spaces and hash)
    text = re.sub(r"[^a-zA-Z#]", " ", text)
    
    # Tokenize and Lemmatize (but DO NOT filter by length)
    tokens = text.split()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(lemmatized)

def main():
    print("Loading dataset...")
    df = pd.read_csv("data1.csv")
    
    # Fix labels
    df['label'] = df['label'].apply(lambda x: 1 if x == -1 else x)
    
    # Double the dataset to match original author's intent (they used combined_df = pd.concat([df, df]))
    combined_df = pd.concat([df, df], ignore_index=True)
    
    print("Preprocessing text... (This may take a minute)")
    combined_df['tidy_tweet'] = combined_df['headline'].apply(preprocess_text)
    
    # Load stopwords
    with open("stopwords.txt", "r", encoding="utf-8") as file:
        stop_words_list = file.read().splitlines()
        
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df['tidy_tweet'], 
        combined_df['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    print("Vectorizing text...")
    # Initialize and FIT vectorizer ONLY on training data
    vectorizer = TfidfVectorizer(stop_words=stop_words_list, lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train.values.astype('U'))
    X_test_vec = vectorizer.transform(X_test.values.astype('U'))
    
    print("Training SGDClassifier...")
    model = SGDClassifier(random_state=42)
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("-" * 40)
    print("ML Pipeline Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("-" * 40)
    
    print("Saving model and vectorizer...")
    os.makedirs("models", exist_ok=True)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open('models/SGDClassifier_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Done! ML Pipeline is fixed and saved.")

if __name__ == "__main__":
    main()
