"""
Shared test fixtures and utilities for the RAG system test suite.
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from fastapi.testclient import TestClient

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from models import Course, Lesson, CourseChunk


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration with temporary paths"""
    config = Config()
    config.CHROMA_PATH = temp_dir
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "test-model"
    config.EMBEDDING_MODEL = "test-embedding-model"
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction"),
            Lesson(lesson_number=2, title="Advanced Topics")
        ]
    )


@pytest.fixture
def sample_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is the introduction to the test course",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This covers advanced topics in the test course",
            course_title="Test Course", 
            lesson_number=2,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_rag_components():
    """Create mocked RAG system components for testing"""
    with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
         patch('rag_system.VectorStore') as mock_vector_store, \
         patch('rag_system.AIGenerator') as mock_ai_gen, \
         patch('rag_system.SessionManager') as mock_session_mgr, \
         patch('rag_system.CourseSearchTool') as mock_search_tool, \
         patch('rag_system.CourseOutlineTool') as mock_outline_tool, \
         patch('rag_system.ToolManager') as mock_tool_manager:
        
        components = {
            'doc_processor': Mock(),
            'vector_store': Mock(),
            'ai_generator': Mock(),
            'session_manager': Mock(),
            'search_tool': Mock(),
            'outline_tool': Mock(),
            'tool_manager': Mock()
        }
        
        # Configure mock constructors to return our mocks
        mock_doc_proc.return_value = components['doc_processor']
        mock_vector_store.return_value = components['vector_store']
        mock_ai_gen.return_value = components['ai_generator']
        mock_session_mgr.return_value = components['session_manager']
        mock_search_tool.return_value = components['search_tool']
        mock_outline_tool.return_value = components['outline_tool']
        mock_tool_manager.return_value = components['tool_manager']
        
        yield components


@pytest.fixture
def test_app():
    """Create a test FastAPI app that doesn't mount static files"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    
    # Create test app without static file mounting
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Mock RAG system for testing
    mock_rag_system = Mock()
    
    # API Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return {"message": "Session cleared successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System Test API"}
    
    # Attach mock for testing
    app.state.rag_system = mock_rag_system
    
    return app


@pytest.fixture
def client(test_app):
    """Create FastAPI test client"""
    return TestClient(test_app)


@pytest.fixture
def mock_query_response():
    """Standard mock response for query testing"""
    return {
        "answer": "This is a test response about machine learning concepts.",
        "sources": [
            {
                "text": "Machine Learning Course - Lesson 1: Introduction",
                "link": "http://example.com/ml/lesson1",
                "course_title": "Machine Learning Course",
                "lesson_number": 1
            },
            {
                "text": "AI Fundamentals - Lesson 3: Neural Networks", 
                "link": "http://example.com/ai/lesson3",
                "course_title": "AI Fundamentals",
                "lesson_number": 3
            }
        ],
        "session_id": "test_session_123"
    }


@pytest.fixture 
def mock_course_analytics():
    """Standard mock course analytics for testing"""
    return {
        "total_courses": 3,
        "course_titles": [
            "Machine Learning Course",
            "AI Fundamentals", 
            "Data Science Basics"
        ]
    }


class TestUtilities:
    """Utility class with common test helper methods"""
    
    @staticmethod
    def create_test_course(title: str = "Test Course", instructor: str = "Test Instructor") -> Course:
        """Helper to create test course objects"""
        return Course(
            title=title,
            instructor=instructor,
            lessons=[
                Lesson(lesson_number=1, title="Introduction"),
                Lesson(lesson_number=2, title="Advanced Topics")
            ]
        )
    
    @staticmethod
    def create_test_chunks(course_title: str = "Test Course", count: int = 2) -> List[CourseChunk]:
        """Helper to create test course chunks"""
        return [
            CourseChunk(
                content=f"Test content chunk {i}",
                course_title=course_title,
                lesson_number=1,
                chunk_index=i
            )
            for i in range(count)
        ]
    
    @staticmethod
    def assert_valid_query_response(response_data: Dict[str, Any]):
        """Helper to validate query response structure"""
        assert "answer" in response_data
        assert "sources" in response_data
        assert "session_id" in response_data
        assert isinstance(response_data["answer"], str)
        assert isinstance(response_data["sources"], list)
        assert isinstance(response_data["session_id"], str)
    
    @staticmethod
    def assert_valid_course_stats(response_data: Dict[str, Any]):
        """Helper to validate course stats response structure"""
        assert "total_courses" in response_data
        assert "course_titles" in response_data
        assert isinstance(response_data["total_courses"], int)
        assert isinstance(response_data["course_titles"], list)


@pytest.fixture
def test_utils():
    """Provide test utilities for convenience"""
    return TestUtilities


# Pytest markers for organizing tests
pytestmark = pytest.mark.unit