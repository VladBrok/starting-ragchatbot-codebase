"""
API endpoint tests for the Course Materials RAG System FastAPI application.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import status


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for the /api/query endpoint"""

    def test_query_endpoint_success_without_session(self, client, mock_query_response, test_utils):
        """Test successful query without providing session_id"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.create_session.return_value = "new_session_456"
        rag_system.query.return_value = (
            mock_query_response["answer"],
            mock_query_response["sources"]
        )
        
        query_data = {"query": "What is machine learning?"}
        
        # Act
        response = client.post("/api/query", json=query_data)
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        test_utils.assert_valid_query_response(response_data)
        assert response_data["answer"] == mock_query_response["answer"]
        assert response_data["sources"] == mock_query_response["sources"]
        assert response_data["session_id"] == "new_session_456"
        
        # Verify RAG system was called correctly
        rag_system.session_manager.create_session.assert_called_once()
        rag_system.query.assert_called_once_with("What is machine learning?", "new_session_456")

    def test_query_endpoint_success_with_session(self, client, mock_query_response):
        """Test successful query with provided session_id"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.query.return_value = (
            mock_query_response["answer"],
            mock_query_response["sources"]
        )
        
        query_data = {
            "query": "Explain neural networks",
            "session_id": "existing_session_789"
        }
        
        # Act
        response = client.post("/api/query", json=query_data)
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["answer"] == mock_query_response["answer"]
        assert response_data["sources"] == mock_query_response["sources"]
        assert response_data["session_id"] == "existing_session_789"
        
        # Verify session creation was not called
        rag_system.session_manager.create_session.assert_not_called()
        rag_system.query.assert_called_once_with("Explain neural networks", "existing_session_789")

    def test_query_endpoint_with_empty_sources(self, client):
        """Test query endpoint when no sources are found"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.create_session.return_value = "session_no_sources"
        rag_system.query.return_value = (
            "I couldn't find specific information about that topic.",
            []  # Empty sources
        )
        
        query_data = {"query": "Obscure topic not in courses"}
        
        # Act
        response = client.post("/api/query", json=query_data)
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["answer"] == "I couldn't find specific information about that topic."
        assert response_data["sources"] == []
        assert response_data["session_id"] == "session_no_sources"

    def test_query_endpoint_rag_system_exception(self, client):
        """Test query endpoint when RAG system raises exception"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.create_session.return_value = "error_session"
        rag_system.query.side_effect = Exception("RAG system error")
        
        query_data = {"query": "This will cause an error"}
        
        # Act
        response = client.post("/api/query", json=query_data)
        
        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "detail" in response_data
        assert "RAG system error" in response_data["detail"]

    def test_query_endpoint_invalid_request_body(self, client):
        """Test query endpoint with invalid request body"""
        # Act - missing required 'query' field
        response = client.post("/api/query", json={"session_id": "test"})
        
        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_empty_query(self, client, mock_query_response):
        """Test query endpoint with empty query string"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.create_session.return_value = "empty_query_session"
        rag_system.query.return_value = (
            "Please provide a specific question about the course materials.",
            []
        )
        
        query_data = {"query": ""}
        
        # Act
        response = client.post("/api/query", json=query_data)
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "Please provide a specific question" in response_data["answer"]

    def test_query_endpoint_long_query(self, client, mock_query_response):
        """Test query endpoint with very long query"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.create_session.return_value = "long_query_session"
        rag_system.query.return_value = (
            mock_query_response["answer"],
            mock_query_response["sources"]
        )
        
        long_query = "What is machine learning? " * 100  # Very long query
        query_data = {"query": long_query}
        
        # Act
        response = client.post("/api/query", json=query_data)
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        rag_system.query.assert_called_once_with(long_query, "long_query_session")


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for the /api/courses endpoint"""

    def test_courses_endpoint_success(self, client, mock_course_analytics, test_utils):
        """Test successful retrieval of course statistics"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.get_course_analytics.return_value = mock_course_analytics
        
        # Act
        response = client.get("/api/courses")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        test_utils.assert_valid_course_stats(response_data)
        assert response_data["total_courses"] == mock_course_analytics["total_courses"]
        assert response_data["course_titles"] == mock_course_analytics["course_titles"]
        
        rag_system.get_course_analytics.assert_called_once()

    def test_courses_endpoint_empty_courses(self, client):
        """Test courses endpoint when no courses exist"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        # Act
        response = client.get("/api/courses")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["total_courses"] == 0
        assert response_data["course_titles"] == []

    def test_courses_endpoint_large_dataset(self, client):
        """Test courses endpoint with large number of courses"""
        # Arrange
        rag_system = client.app.state.rag_system
        large_course_list = [f"Course {i}" for i in range(100)]
        rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": large_course_list
        }
        
        # Act
        response = client.get("/api/courses")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["total_courses"] == 100
        assert len(response_data["course_titles"]) == 100
        assert response_data["course_titles"][0] == "Course 0"
        assert response_data["course_titles"][-1] == "Course 99"

    def test_courses_endpoint_rag_system_exception(self, client):
        """Test courses endpoint when RAG system raises exception"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.get_course_analytics.side_effect = Exception("Analytics retrieval failed")
        
        # Act
        response = client.get("/api/courses")
        
        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "detail" in response_data
        assert "Analytics retrieval failed" in response_data["detail"]


@pytest.mark.api
class TestSessionEndpoint:
    """Test suite for the /api/session/{session_id} endpoint"""

    def test_delete_session_success(self, client):
        """Test successful session deletion"""
        # Arrange
        rag_system = client.app.state.rag_system
        session_id = "session_to_delete"
        
        # Act
        response = client.delete(f"/api/session/{session_id}")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        
        assert response_data["message"] == "Session cleared successfully"
        rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_delete_session_with_special_characters(self, client):
        """Test session deletion with special characters in session ID"""
        # Arrange
        rag_system = client.app.state.rag_system
        session_id = "session-with_special.chars123"
        
        # Act
        response = client.delete(f"/api/session/{session_id}")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_delete_session_nonexistent(self, client):
        """Test deletion of nonexistent session"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.clear_session.side_effect = Exception("Session not found")
        session_id = "nonexistent_session"
        
        # Act
        response = client.delete(f"/api/session/{session_id}")
        
        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "Session not found" in response_data["detail"]

    def test_delete_session_session_manager_exception(self, client):
        """Test session deletion when session manager raises exception"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.clear_session.side_effect = Exception("Database connection error")
        session_id = "error_session"
        
        # Act
        response = client.delete(f"/api/session/{session_id}")
        
        # Assert
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "Database connection error" in response_data["detail"]


@pytest.mark.api
class TestRootEndpoint:
    """Test suite for the root endpoint /"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns proper message"""
        # Act
        response = client.get("/")
        
        # Assert
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert "message" in response_data
        assert "Course Materials RAG System Test API" in response_data["message"]


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API workflow scenarios"""

    def test_complete_query_session_workflow(self, client, mock_query_response):
        """Test complete workflow: query -> courses -> delete session"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.session_manager.create_session.return_value = "workflow_session"
        rag_system.query.return_value = (
            mock_query_response["answer"],
            mock_query_response["sources"]
        )
        rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course 1", "Course 2"]
        }
        
        # Act & Assert - Step 1: Query
        query_response = client.post("/api/query", json={"query": "Test query"})
        assert query_response.status_code == status.HTTP_200_OK
        session_id = query_response.json()["session_id"]
        
        # Act & Assert - Step 2: Get courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK
        assert courses_response.json()["total_courses"] == 2
        
        # Act & Assert - Step 3: Delete session
        delete_response = client.delete(f"/api/session/{session_id}")
        assert delete_response.status_code == status.HTTP_200_OK
        assert "cleared successfully" in delete_response.json()["message"]

    def test_multiple_queries_same_session(self, client, mock_query_response):
        """Test multiple queries with the same session ID"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.query.side_effect = [
            ("First response", mock_query_response["sources"]),
            ("Second response with context", mock_query_response["sources"])
        ]
        session_id = "persistent_session"
        
        # Act - First query
        response1 = client.post("/api/query", json={
            "query": "First question",
            "session_id": session_id
        })
        
        # Act - Second query with same session
        response2 = client.post("/api/query", json={
            "query": "Follow-up question", 
            "session_id": session_id
        })
        
        # Assert
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK
        
        assert response1.json()["answer"] == "First response"
        assert response2.json()["answer"] == "Second response with context"
        
        # Both should have same session ID
        assert response1.json()["session_id"] == session_id
        assert response2.json()["session_id"] == session_id
        
        # Verify RAG system was called correctly for both
        assert rag_system.query.call_count == 2
        rag_system.query.assert_any_call("First question", session_id)
        rag_system.query.assert_any_call("Follow-up question", session_id)

    def test_concurrent_sessions(self, client, mock_query_response):
        """Test handling of multiple concurrent sessions"""
        # Arrange
        rag_system = client.app.state.rag_system
        rag_system.query.return_value = (
            mock_query_response["answer"],
            mock_query_response["sources"]
        )
        
        session1 = "concurrent_session_1"
        session2 = "concurrent_session_2"
        
        # Act - Queries from different sessions
        response1 = client.post("/api/query", json={
            "query": "Question from session 1",
            "session_id": session1
        })
        
        response2 = client.post("/api/query", json={
            "query": "Question from session 2", 
            "session_id": session2
        })
        
        # Assert
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK
        
        assert response1.json()["session_id"] == session1
        assert response2.json()["session_id"] == session2
        
        # Verify both sessions were handled
        rag_system.query.assert_any_call("Question from session 1", session1)
        rag_system.query.assert_any_call("Question from session 2", session2)


@pytest.mark.api
class TestAPIErrorHandling:
    """Test comprehensive error handling scenarios"""

    def test_malformed_json_request(self, client):
        """Test API handling of malformed JSON requests"""
        # Act
        response = client.post(
            "/api/query",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        
        # Assert
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_content_type(self, client):
        """Test API handling when Content-Type is missing"""
        # Act
        response = client.post("/api/query", data='{"query": "test"}')
        
        # Assert - FastAPI may return 500 if RAG system fails due to malformed request
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY, 
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    def test_unsupported_http_methods(self, client):
        """Test unsupported HTTP methods on endpoints"""
        # Test unsupported methods on query endpoint
        response_put = client.put("/api/query", json={"query": "test"})
        assert response_put.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
        
        response_patch = client.patch("/api/query", json={"query": "test"})
        assert response_patch.status_code == status.HTTP_405_METHOD_NOT_ALLOWED
        
        # Test unsupported methods on courses endpoint
        response_post = client.post("/api/courses", json={})
        assert response_post.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_nonexistent_endpoints(self, client):
        """Test requests to nonexistent endpoints"""
        # Act
        response = client.get("/api/nonexistent")
        
        # Assert
        assert response.status_code == status.HTTP_404_NOT_FOUND


if __name__ == "__main__":
    pytest.main([__file__, "-v"])