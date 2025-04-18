import datetime as dt
from fastapi import FastAPI
from pymongo import MongoClient
from core.config import settings
from user.views import user_views
from spaces.views import space_views
from patients.views import patient_views
from agent.views import agent_views
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],
)

# Global variable to store the MongoDB client
client = None

@app.on_event("startup")
async def connect_to_db():
    global client
    client = MongoClient(settings.mongo_uri)
    print("Connected to the database")
    

@app.on_event("shutdown")
async def close_db_connection():
    global client
    if client:
        client.close()
        print("Database connection closed")

@app.get("/")
async def read_root():
    return {
        "Application": "Responsible AI API",
        "Purpose": "This is a project for FastAPI with MongoDB.",
    }

app.include_router(user_views.router, prefix="/user")
app.include_router(space_views.router, prefix="/space")
app.include_router(patient_views.router, prefix="/patient")
app.include_router(agent_views.router, prefix="/agent")
