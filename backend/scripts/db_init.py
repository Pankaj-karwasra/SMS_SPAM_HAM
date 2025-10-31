import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from database.connection import engine, Base
from repository.prediction_repo import Prediction 

def init_db():
    print("Attempting to create database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully (or already exist).")

if __name__ == "__main__":
    init_db()