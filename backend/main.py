from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.models.prediction_models import SMSMessage, PredictionResult, PredictionDB
from backend.services.ml_pipeline import load_ml_artifacts, run_prediction_pipeline
from backend.database.connection import get_db
from backend.services.db_service import DBService
from backend.core.config import settings
from backend.core.exceptions import ModelLoadingError, PredictionProcessingError, DatabaseError
import os

# Import StaticFiles
from fastapi.staticfiles import StaticFiles 

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="An API to classify SMS messages as Ham or Spam.",
    version="1.0.0"
)

# --- CORS Middleware for Frontend ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FastAPI Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Load ML models when the FastAPI application starts."""
    print("Application startup: Loading ML artifacts...")
    try:
        load_ml_artifacts()
    except Exception as e:
        print(f"Failed to load ML artifacts at startup: {e}")
        raise ModelLoadingError(detail=f"Failed to load ML models at startup: {e}")


@app.post("/predict", response_model=PredictionResult, status_code=status.HTTP_200_OK)
async def predict(sms: SMSMessage, db: Session = Depends(get_db)):
    """
    Classifies an incoming SMS message as 'ham' (0) or 'spam' (1)
    and logs the prediction to the database.
    """
    try:
        prediction_label = run_prediction_pipeline(sms.message)
        result_str = "spam" if prediction_label == 1 else "ham"
    except RuntimeError as e:
        raise ModelLoadingError(detail=str(e))
    except Exception as e:
        print(f"Prediction processing error for message '{sms.message}': {e}")
        raise PredictionProcessingError(detail=f"Error during message classification: {e}")

    try:
        db_service = DBService(db)
        created_prediction = db_service.create_and_log_prediction(sms.message, prediction_label)

        return PredictionResult(
            original_message=sms.message,
            prediction=result_str,
            label_encoded=prediction_label,
            timestamp=created_prediction.timestamp,
            record_id=created_prediction.id
        )
    except Exception as e:
        print(f"Database error while logging prediction for message '{sms.message}': {e}")
        raise DatabaseError(detail=f"Failed to log prediction to database: {e}")

@app.get("/predictions", response_model=list[PredictionDB], status_code=status.HTTP_200_OK)
async def get_all_predictions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieves a list of all logged predictions from the database.
    (Admin/Audit endpoint)
    """
    try:
        db_service = DBService(db)
        predictions = db_service.get_all_predictions(skip=skip, limit=limit)
        return predictions
    except Exception as e:
        print(f"Database error while retrieving predictions: {e}")
        raise DatabaseError(detail=f"Failed to retrieve predictions from database: {e}")


script_dir = os.path.dirname(__file__)

frontend_public_dir = os.path.join(script_dir, os.pardir, 'frontend', 'public')


from starlette.responses import FileResponse 

@app.get("/", include_in_schema=False) 
async def serve_frontend():
    index_html_path = os.path.join(frontend_public_dir, 'index.html')
    if not os.path.exists(index_html_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_html_path)


app.mount("/", StaticFiles(directory=frontend_public_dir, html=True), name="frontend_static")