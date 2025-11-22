# PreMed RAG Application

A FastAPI-based RAG (Retrieval-Augmented Generation) application for querying premed questions using OpenAI embeddings and Qdrant vector database.

## Features

- Load JSON files and create embeddings using OpenAI `text-embeddings-3-large`
- Store embeddings in Qdrant vector database
- Query questions with similarity threshold filtering (default 75%)
- Industry-standard project architecture with separated concerns

## Project Structure

```
premed-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic models for API
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding_service.py    # OpenAI embedding service
│   │   └── vector_store_service.py # Qdrant vector store service
│   └── api/
│       ├── __init__.py
│       └── routes.py          # FastAPI routes
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Start Qdrant (using Docker):**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Running the Application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

### 1. Load JSON File
**POST** `/api/v1/load-json`

Load a JSON file and create embeddings for all objects.

**Request Body:**
```json
{
  "json_file_path": "/path/to/your/questions.json"
}
```

**Response:**
```json
{
  "message": "Successfully loaded 100 objects from JSON file",
  "total_objects": 100,
  "collection_name": "premed_questions"
}
```

### 2. Query Questions
**POST** `/api/v1/query`

Query for relevant questions based on similarity.

**Request Body:**
```json
{
  "query": "What is the mechanism of action of aspirin?",
  "threshold": 0.75,
  "limit": 10
}
```

**Response:**
```json
{
  "query": "What is the mechanism of action of aspirin?",
  "results": [
    {
      "id": "uuid-here",
      "question": {
        "question": "How does aspirin work?",
        "answer": "..."
      },
      "score": 0.89
    }
  ],
  "total_results": 1,
  "threshold": 0.75
}
```

### 3. Health Check
**GET** `/api/v1/health`

Check if the service is running.

## JSON File Format

The JSON file should be an array of objects. Each object will be converted to text for embedding. If an object has a `question` field, it will be prioritized. Otherwise, all fields will be concatenated.

Example:
```json
[
  {
    "question": "What is the function of the heart?",
    "answer": "The heart pumps blood throughout the body.",
    "category": "anatomy"
  },
  {
    "question": "What is DNA?",
    "answer": "DNA is the genetic material.",
    "category": "biology"
  }
]
```

## Configuration

All configuration is done through environment variables (see `.env.example`):

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `QDRANT_HOST`: Qdrant host (default: localhost)
- `QDRANT_PORT`: Qdrant port (default: 6333)
- `QDRANT_COLLECTION_NAME`: Collection name (default: premed_questions)
- `SIMILARITY_THRESHOLD`: Default similarity threshold (default: 0.75)

