from sqlalchemy.orm import Session
from backend.repository.prediction_repo import PredictionRepository, Prediction

class DBService:
    def __init__(self, db: Session):
        self.db = db
        self.prediction_repo = PredictionRepository(db)

    def create_and_log_prediction(self, message: str, prediction_label: int) -> Prediction:
        """
        Logs a new prediction to the database.
        """
        return self.prediction_repo.create_prediction(message, prediction_label)

    def get_all_predictions(self, skip: int = 0, limit: int = 100) -> list[Prediction]:
        """
        Retrieves all logged predictions.
        """
        return self.prediction_repo.get_predictions(skip=skip, limit=limit)