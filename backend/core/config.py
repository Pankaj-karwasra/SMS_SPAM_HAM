from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
   
    app_name: str = "SMS Spam Classifier API"
    
    # Database settings
    db_url: str = "sqlite:///./sql_app.db" 
    
    model_path: str = "ml_artifacts/mnb_model.pkl" 
    vectorizer_path: str = "ml_artifacts/tfidf_vectorizer.pkl"

    model_config = SettingsConfigDict(env_file='.env', extra='ignore') 
settings = Settings()


if 'NLTK_DATA' not in os.environ:
    pass