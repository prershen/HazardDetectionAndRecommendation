# Responsible AI Backend Server

This repository contains the backend server for the Responsible AI application, built with FastAPI and MongoDB.

## Prerequisites

- Python 3.9+
- MongoDB (or access to MongoDB Atlas)
- Git

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
