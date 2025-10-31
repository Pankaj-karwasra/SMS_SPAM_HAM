from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SMSMessage(BaseModel):
    message: str = Field(..., description="The SMS message to classify.", example="Free money now! Click here!")

class PredictionResult(BaseModel):
    original_message: str = Field(..., description="The original message that was classified.")
    prediction: str = Field(..., description="The classification result: 'ham' or 'spam'.", example="spam")
    label_encoded: int = Field(..., description="The numerical encoded label: 0 for ham, 1 for spam.", example=1)
    timestamp: datetime = Field(..., description="Timestamp when the prediction was recorded.")
    record_id: int = Field(..., description="Unique ID of the prediction record in the database.")

class PredictionBase(BaseModel):
    message: str
    prediction: int

class PredictionCreate(PredictionBase):
    pass 

class PredictionDB(PredictionBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True 