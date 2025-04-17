from fastapi import Depends
from pymongo import MongoClient
from pymongo.collection import Collection
from core.config import settings
import main  # To access the global client variable

def get_db():
    client = main.client
    db = client[settings.database_name]
    return db
