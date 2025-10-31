import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import os

def download_nltk_data():
    """Ensures necessary NLTK data is downloaded."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

def load_data(filepath=None):
    """Loads the SMS Spam Collection dataset."""
    if filepath is None:
        script_dir = os.path.dirname(__file__)
     
        parent_dir_of_script = os.path.abspath(os.path.join(script_dir, os.pardir))
        filepath = os.path.join(parent_dir_of_script, 'data', 'SMSSpamCollection')
    
    print(f"Attempting to load data from: {filepath}")

    df = pd.read_csv(filepath, sep='\t', names=['label', 'message'])
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Applies consistent text preprocessing steps."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    processed_words = [
        _stemmer.stem(word) for word in words if word not in _stop_words and word.isalpha()
    ]
    return " ".join(processed_words)

def visualize_data(df):
    """Generates visualizations for data distribution."""
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Distribution of Ham vs. Spam Messages')
    plt.xlabel('Message Type')
    plt.ylabel('Count')
    plt.show()

def evaluate_model(y_test, y_pred):
    """Prints and visualizes model evaluation metrics."""
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, pos_label=1):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, pos_label=1):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Ham', 'Predicted Spam'],
                yticklabels=['Actual Ham', 'Actual Spam'])
    plt.title('Confusion Matrix for SMS Spam Classifier')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    print("--- Starting Model Training ---")

    download_nltk_data()

    print("1. Loading and cleaning data...")
    df = load_data() 
    print("First 5 rows:\n", df.head())
    print("\nDataset Info:")
    df.info()
    print(f"\nDataset Shape: {df.shape}")
    print("\nMissing values per column:\n", df.isnull().sum())
    visualize_data(df)

    print("\n2. Preprocessing text messages...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    print("First 5 rows with processed messages:\n", df[['message', 'processed_message', 'label_encoded']].head())

    print("\n3. Feature Extraction using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    X = tfidf_vectorizer.fit_transform(df['processed_message'])
    y = df['label_encoded']
    print(f"Shape of the TF-IDF feature matrix (X): {X.shape}")
    print(f"Shape of the target variable (y): {y.shape}")

    print("\n4. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    print("\n5. Training Naive Bayes Classifier...")
    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(X_train, y_train)
    print("Multinomial Naive Bayes Classifier trained successfully!")

    print("\n6. Making predictions on the test set...")
    y_pred = mnb_classifier.predict(X_test)
    print("\nSample of Actual vs. Predicted Labels on Test Set:\n", pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(10))

    print("\n7. Evaluating model performance...")
    evaluate_model(y_test, y_pred)

    print("\n8. Saving trained model and TF-IDF vectorizer...")
    script_dir = os.path.dirname(__file__)

    parent_dir_of_script = os.path.abspath(os.path.join(script_dir, os.pardir))
    ml_artifacts_dir = os.path.join(parent_dir_of_script, 'ml_artifacts')
    os.makedirs(ml_artifacts_dir, exist_ok=True)

    with open(os.path.join(ml_artifacts_dir, 'mnb_model.pkl'), 'wb') as f:
        pickle.dump(mnb_classifier, f)
    with open(os.path.join(ml_artifacts_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"Multinomial Naive Bayes classifier and TF-IDF vectorizer saved to '{ml_artifacts_dir}/'")

    print("\n--- Model Training Complete ---")