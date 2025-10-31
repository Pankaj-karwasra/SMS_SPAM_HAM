Features

✅ Machine Learning Pipeline Integration –
Includes a complete ML pipeline built using scikit-learn that handles text preprocessing, vectorization (TF-IDF/CountVectorizer), and classification.
The pipeline is loaded at FastAPI startup using load_ml_artifacts() from backend/services/ml_pipeline.py, and predictions are made through run_prediction_pipeline().

✅ Database Logging –
Each prediction (SMS message, label, timestamp) is automatically stored in the database using SQLAlchemy models and DBService.

✅ FastAPI Framework –
Built with FastAPI, offering async performance, built-in data validation via Pydantic models, and auto-generated API documentation (Swagger UI and ReDoc).

✅ CORS Enabled –
CORS middleware is configured to allow cross-origin requests from local or remote frontends, enabling seamless integration with a React or other JS frontend.

✅ Frontend Hosting –
FastAPI also serves the static frontend from /frontend/public, allowing a unified backend + frontend deployment.

✅ Error Handling –
Includes custom exception classes:

ModelLoadingError → Raised when ML models fail to load

PredictionProcessingError → Raised during model inference errors

DatabaseError → Raised when database operations fail
