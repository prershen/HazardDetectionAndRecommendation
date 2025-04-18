import datetime as dt
from fastapi import FastAPI, HTTPException, Depends
from pymongo import MongoClient
from core.config import settings
from user.views import user_views
from spaces.views import space_views
from patients.views import patient_views
from fastapi.middleware.cors import CORSMiddleware
from patients.db.indexes import create_patient_indexes
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load environment variables
load_dotenv()

app = FastAPI(title="Llama Chatbot API")

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

# MongoDB connection
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["chatbot_db"]
spaces_collection = db["spaces"]
conversations_collection = db["conversations"]

# Load Llama model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load LlamaGuard
guard_model_name = "meta-llama/LlamaGuard-7b"
guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_name)
guard_model = AutoModelForCausalLM.from_pretrained(guard_model_name)

class Space(BaseModel):
    name: str
    description: str
    context: str

class Message(BaseModel):
    content: str
    role: str  # "user" or "assistant"

class Conversation(BaseModel):
    space_id: str
    messages: List[Message]

@app.on_event("startup")
async def connect_to_db():
    global client
    client = MongoClient(settings.mongo_uri)
    print("Connected to the database")
    
    # Create indexes
    db = client[settings.database_name]
    create_patient_indexes(db)

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

@app.post("/spaces/")
async def create_space(space: Space):
    space_dict = space.dict()
    result = spaces_collection.insert_one(space_dict)
    return {"id": str(result.inserted_id), **space_dict}

@app.get("/spaces/")
async def get_spaces():
    spaces = list(spaces_collection.find({}, {"_id": 0}))
    return spaces

@app.post("/chat/{space_id}")
async def chat(space_id: str, message: str):
    # Get space context
    space = spaces_collection.find_one({"_id": space_id})
    if not space:
        raise HTTPException(status_code=404, detail="Space not found")
    
    # Prepare prompt with context
    prompt = f"Context: {space['context']}\nUser: {message}\nAssistant:"
    
    # Generate response with Llama
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Validate response with LlamaGuard
    guard_prompt = f"Is this response appropriate and accurate? Response: {response}"
    guard_inputs = guard_tokenizer(guard_prompt, return_tensors="pt")
    guard_outputs = guard_model.generate(**guard_inputs, max_length=50)
    guard_response = tokenizer.decode(guard_outputs[0], skip_special_tokens=True)
    
    # If LlamaGuard flags the response, generate a safer response
    if "unsafe" in guard_response.lower():
        response = "I apologize, but I cannot provide that information. Please ask something else."
    
    # Store conversation
    conversation = {
        "space_id": space_id,
        "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
    }
    conversations_collection.insert_one(conversation)
    
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
