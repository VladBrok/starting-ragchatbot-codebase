# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Course Materials RAG (Retrieval-Augmented Generation) System that provides intelligent, context-aware responses to questions about course materials. It's a full-stack Python application using FastAPI for the backend, vanilla HTML/CSS/JavaScript for the frontend, and ChromaDB for vector storage.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Dependency Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>
```

### Environment Setup
Create a `.env` file in the root directory with:
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Architecture Overview

### Core Components
- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`): ChromaDB wrapper for semantic search and storage  
- **DocumentProcessor** (`backend/document_processor.py`): Processes course documents into chunks
- **AIGenerator** (`backend/ai_generator.py`): Handles Anthropic Claude API interactions with tool support
- **SessionManager** (`backend/session_manager.py`): Manages conversation history
- **ToolManager** (`backend/search_tools.py`): Provides search tools for the AI agent

### Data Models (`backend/models.py`)
- **Course**: Represents a complete course with lessons
- **Lesson**: Individual lesson within a course
- **CourseChunk**: Text chunks for vector storage

### API Structure
- **FastAPI App** (`backend/app.py`): 
  - `/api/query` - Process queries and return AI responses with sources
  - `/api/courses` - Get course analytics and statistics
  - Static file serving for frontend at `/`

### Frontend (`frontend/`)
- Simple HTML/CSS/JavaScript interface
- No build process required - served as static files

## Key Implementation Details

### Vector Storage Strategy
- Two ChromaDB collections:
  - `course_catalog`: Course metadata for semantic course name resolution
  - `course_content`: Actual course content chunks for retrieval

### Tool-Based Search
The system uses Claude's tool calling capabilities rather than traditional RAG context injection:
- AI can call search tools to find relevant information
- More flexible and allows for multiple search operations per query
- Sources are tracked through tool execution

### Document Processing
- Supports PDF, DOCX, and TXT files
- Documents are chunked with configurable size (800 chars) and overlap (100 chars)
- Course titles used as unique identifiers

### Configuration (`backend/config.py`)
- Environment-based configuration using python-dotenv
- Key settings: embedding model, chunk sizes, API keys, database paths

## Development Notes

- No test framework currently configured
- No linting tools configured in project
- Uses `uv` for Python package management instead of pip/poetry
- ChromaDB data persisted in `./chroma_db` directory
- Development server includes no-cache headers for frontend static files
- CORS enabled for development with permissive settings
- the server is always running by default, no need to launch it. after code changes are made, the server automatically restarts

# ultrathink
