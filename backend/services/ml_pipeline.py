
import pickle
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB 

_tfidf_vectorizer = None
_mnb_classifier = None
_stemmer = None
_stop_words = None 
_script_dir = os.path.dirname(__file__)

_backend_dir = os.path.abspath(os.path.join(_script_dir, os.pardir))

_ml_artifacts_base_path = os.path.join(_backend_dir, 'ml_artifacts')


def load_ml_artifacts():
    global _tfidf_vectorizer, _mnb_classifier, _stemmer, _stop_words

    vectorizer_path = os.path.join(_ml_artifacts_base_path, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(_ml_artifacts_base_path, 'mnb_model.pkl')

    print(f"Attempting to load TF-IDF from: {vectorizer_path}")
    print(f"Attempting to load MNB model from: {model_path}")

    if not os.path.exists(vectorizer_path):
        raise RuntimeError(f"TF-IDF vectorizer not found at: {vectorizer_path}")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Multinomial Naive Bayes model not found at: {model_path}")

    try:
        with open(vectorizer_path, 'rb') as f:
            _tfidf_vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            _mnb_classifier = pickle.load(f)
        
        _stemmer = PorterStemmer()
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        _stop_words = set(stopwords.words('english'))

        print("ML artifacts loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Error loading ML artifacts: {e}")

def _preprocess_text_for_pipeline(text: str) -> str:
    """Helper function for consistent preprocessing within the prediction pipeline."""
    if _stemmer is None or _stop_words is None:
        raise RuntimeError("Preprocessing resources (stemmer/stopwords) not initialized.")

    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    processed_words = [
        _stemmer.stem(word) for word in words if word not in _stop_words and word.isalpha()
    ]
    return " ".join(processed_words)


def run_prediction_pipeline(message: str) -> int:
    """
    Runs the full ML pipeline for a given message to predict if it's ham (0) or spam (1).
    """
    if _tfidf_vectorizer is None or _mnb_classifier is None:
        raise RuntimeError("ML models are not loaded. Call load_ml_artifacts() first.")

    processed_message = _preprocess_text_for_pipeline(message)

    message_tfidf = _tfidf_vectorizer.transform([processed_message])
    prediction = _mnb_classifier.predict(message_tfidf)[0]

    return int(prediction) 