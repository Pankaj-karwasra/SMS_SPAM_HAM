import os
import sys
from sqlalchemy import create_engine

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

sys.path.insert(0, project_root)


from backend.database.connection import Base, SQLALCHEMY_DATABASE_URL
from backend.repository.prediction_repo import Prediction 

def init_db():
    print(f"Initializing database at: {SQLALCHEMY_DATABASE_URL}")
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    print("Database tables created (if they didn't already exist).")

if __name__ == "__main__":
    if "sqlite:///" in SQLALCHEMY_DATABASE_URL:
        db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"Created database directory: {db_dir}")

    init_db()