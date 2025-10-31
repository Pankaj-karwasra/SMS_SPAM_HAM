from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import Session
from backend.database.connection import Base 

class Prediction(Base):
    __tablename__ = "predictions" 
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, index=True, nullable=False)
    prediction = Column(Integer, nullable=False) 
    timestamp = Column(DateTime(timezone=True), server_default=func.now()) 

class PredictionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_prediction(self, message: str, prediction_label: int) -> Prediction:
        """
        Creates a new prediction record in the database.
        """
        db_prediction = Prediction(message=message, prediction=prediction_label)
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction) 
        return db_prediction
    
    def get_predictions(self, skip: int = 0, limit: int = 100) -> list[Prediction]:
        """
        Retrieves a list of prediction records from the database.
        """
        return self.db.query(Prediction).offset(skip).limit(limit).all()
    
   