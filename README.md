# Responsible AI Backend Server

As international students, we’ve all asked ourselves:

"𝗛𝗼𝘄 𝗰𝗮𝗻 𝘄𝗲 𝗯𝗲 𝗽𝗿𝗲𝘀𝗲𝗻𝘁 𝗳𝗼𝗿 𝘁𝗵𝗲 𝗽𝗲𝗼𝗽𝗹𝗲 𝘄𝗲 𝗹𝗼𝘃𝗲, 𝗲𝘃𝗲𝗻 𝘄𝗵𝗲𝗻 𝘄𝗲'𝗿𝗲……..𝗺𝗶𝗹𝗲𝘀 𝗮𝘄𝗮𝘆?"

That question becomes even more pressing when our loved ones are recovering from surgery, aging, or need at-home care.



That’s where 𝗥𝗶𝘀𝗸-𝗜𝘁-𝗙𝗿𝗲𝗲 comes in. 



🏠 Within 24 hours at the SAIRS Hackathon, we built a home safety app to proactively detect risks in living spaces, generate actionable safety recommendations, and offer empathetic AI support.



Some key highlights are as follows

1️⃣ 𝗣𝗲𝗿𝘀𝗼𝗻𝗮𝗹𝗶𝘇𝗲𝗱 𝗥𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱𝗮𝘁𝗶𝗼𝗻 𝗦𝘆𝘀𝘁𝗲𝗺

Smart, condition-aware recommendations driven by a real-time safety scoring system. We combined object detection with retrieval, embeddings, and Safety Guardrails to ensure adaptive risk prioritization.

2️⃣ 𝗚𝗲𝗻𝗔𝗜-𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗛𝗮𝘇𝗮𝗿𝗱 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻 

We used Google Gemini Flash 2.0 and OWL V2 to detect critical hazards—loose rugs, clutter, wires—with context awareness and grounded object recognition.

3️⃣ 𝗖𝗼𝗻𝘃𝗲𝗿𝘀𝗮𝘁𝗶𝗼𝗻𝗮𝗹 𝗔𝗜 𝗖𝗼𝗺𝗽𝗮𝗻𝗶𝗼𝗻  

Our scalable chatbot, powered by LangGraph agents, RAG, and MongoDB, serves as a safety assistant, answering questions, providing guidance, and fact-checking via GuardRails to maintain responsible AI outputs.

4️⃣ 𝗚𝗮𝗺𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻 𝗮𝗻𝗱 𝗔𝗰𝗰𝗲𝘀𝘀𝗶𝗯𝗶𝗹𝗶𝘁𝘆

 We designed the experience with accessibility-first principles and turned room safety into an engaging challenge by introducing a 𝗦𝗮𝗳𝗲𝘁𝘆 𝗦𝗰𝗼𝗿𝗲 to motivate action.

This repository contains the backend server for the Responsible AI application, built with FastAPI and MongoDB. For the whole project, see https://github.com/risk-it-free. For more information about the hackathon, see https://devpost.com/software/risk-it-free

## Prerequisites

- Python 3.9+
- MongoDB (or access to MongoDB Atlas)
- AWS
- Gemini API key

Install all the requirements provided

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/backend-server.git
cd backend-server
```

### 2. Set Up Environment

You can set up the environment using either pip or conda:

#### Using pip (Virtual Environment)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory with the following variables (or use the existing one):

```
MONGO_URI=your_mongodb_connection_string
DATABASE_NAME=your_database_name
JWT_SECRET=your_jwt_secret
GEMINI_API_KEY=your_gemini_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
BUCKET_NAME=your_bucket_name
```

### 4. Starting the Server

To start the development server:

```bash
# Using uvicorn directly
uvicorn main:app --reload

# OR with specific host and port
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000` by default.

## API Documentation

Once the server is running, you can access:

- API documentation at `http://localhost:8000/docs`
- Alternative documentation at `http://localhost:8000/redoc`

## Project Structure

- `main.py`: Entry point for the FastAPI application
- `core/`: Core configuration and utilities
- `user/`: User management endpoints
- `spaces/`: Spaces management endpoints
- `patients/`: Patient management endpoints
- `simple_agent/`: Agent functionality endpoints

## License

[Your License Information]
