import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from config import Config


class TestRAGSystem:
    """Integration test suite for RAGSystem"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()

        # Create test config
        self.test_config = Config()
        self.test_config.CHROMA_PATH = self.temp_dir
        self.test_config.MAX_RESULTS = 5  # Set to non-zero for testing
        self.test_config.ANTHROPIC_API_KEY = "test-key"
        self.test_config.ANTHROPIC_MODEL = "test-model"

        # Mock all external dependencies
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_proc,
            patch("rag_system.VectorStore") as mock_vector_store,
            patch("rag_system.AIGenerator") as mock_ai_gen,
            patch("rag_system.SessionManager") as mock_session_mgr,
            patch("rag_system.CourseSearchTool") as mock_search_tool,
            patch("rag_system.CourseOutlineTool") as mock_outline_tool,
            patch("rag_system.ToolManager") as mock_tool_manager,
        ):

            # Set up mocks
            self.mock_doc_processor = Mock()
            self.mock_vector_store = Mock()
            self.mock_ai_generator = Mock()
            self.mock_session_manager = Mock()
            self.mock_search_tool = Mock()
            self.mock_outline_tool = Mock()
            self.mock_tool_manager = Mock()

            mock_doc_proc.return_value = self.mock_doc_processor
            mock_vector_store.return_value = self.mock_vector_store
            mock_ai_gen.return_value = self.mock_ai_generator
            mock_session_mgr.return_value = self.mock_session_manager
            mock_search_tool.return_value = self.mock_search_tool
            mock_outline_tool.return_value = self.mock_outline_tool
            mock_tool_manager.return_value = self.mock_tool_manager

            self.rag_system = RAGSystem(self.test_config)

    def teardown_method(self):
        """Clean up after each test method"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_components(self):
        """Test that RAGSystem initializes all components correctly"""
        # Assert that all components were created with correct config
        assert self.rag_system.document_processor == self.mock_doc_processor
        assert self.rag_system.vector_store == self.mock_vector_store
        assert self.rag_system.ai_generator == self.mock_ai_generator
        assert self.rag_system.session_manager == self.mock_session_manager

        # Assert tool manager was set up with both tools
        assert self.mock_tool_manager.register_tool.call_count == 2

    def test_query_successful_flow(self):
        """Test successful end-to-end query processing"""
        # Arrange
        query = "What is machine learning?"
        session_id = "test_session_123"

        # Mock conversation history
        self.mock_session_manager.get_conversation_history.return_value = (
            "Previous: Hello"
        )

        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = (
            "Machine learning is a subset of AI..."
        )

        # Mock tool manager sources
        mock_sources = [
            {"text": "ML Course - Lesson 1", "link": "http://example.com/ml1"},
            {"text": "AI Basics - Lesson 2", "link": "http://example.com/ai2"},
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources

        # Act
        response, sources = self.rag_system.query(query, session_id)

        # Assert
        assert response == "Machine learning is a subset of AI..."
        assert sources == mock_sources

        # Verify proper call sequence
        self.mock_session_manager.get_conversation_history.assert_called_once_with(
            session_id
        )

        expected_prompt = f"Answer this question about course materials: {query}"
        self.mock_ai_generator.generate_response.assert_called_once_with(
            query=expected_prompt,
            conversation_history="Previous: Hello",
            tools=self.mock_tool_manager.get_tool_definitions.return_value,
            tool_manager=self.mock_tool_manager,
        )

        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, query, response
        )

        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()

    def test_query_without_session(self):
        """Test query processing without session ID"""
        # Arrange
        query = "Test query"

        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Test response"
        self.mock_tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        assert response == "Test response"
        assert sources == []

        # Verify no session operations occurred
        self.mock_session_manager.get_conversation_history.assert_not_called()
        self.mock_session_manager.add_exchange.assert_not_called()

        # Verify AI was called without history
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] is None

    def test_query_with_empty_sources(self):
        """Test query when no sources are found"""
        # Arrange
        query = "Obscure topic"

        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = (
            "I couldn't find specific information about that topic."
        )
        self.mock_tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        assert response == "I couldn't find specific information about that topic."
        assert sources == []

        # Verify tool definitions were still provided
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        assert (
            call_args[1]["tools"]
            == self.mock_tool_manager.get_tool_definitions.return_value
        )

    def test_query_critical_max_results_zero_scenario(self):
        """Test query behavior with MAX_RESULTS=0 configuration issue"""
        # Arrange
        self.test_config.MAX_RESULTS = 0  # Simulate the critical issue

        query = "Valid content query"

        # Mock AI response that indicates search failed
        self.mock_ai_generator.generate_response.return_value = "query failed"
        self.mock_tool_manager.get_last_sources.return_value = []

        # Act
        response, sources = self.rag_system.query(query)

        # Assert
        assert response == "query failed"
        assert sources == []

        # This test should help identify if MAX_RESULTS=0 is causing the issue

    def test_add_course_document_success(self):
        """Test successful course document addition"""
        # Arrange
        file_path = "/path/to/course.pdf"

        test_course = Course(
            title="Test Course",
            instructor="Test Instructor",
            lessons=[Lesson(lesson_number=1, title="Intro")],
        )
        test_chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        self.mock_doc_processor.process_course_document.return_value = (
            test_course,
            test_chunks,
        )

        # Act
        course, chunk_count = self.rag_system.add_course_document(file_path)

        # Assert
        assert course == test_course
        assert chunk_count == 1

        # Verify proper call sequence
        self.mock_doc_processor.process_course_document.assert_called_once_with(
            file_path
        )
        self.mock_vector_store.add_course_metadata.assert_called_once_with(test_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(test_chunks)

    def test_add_course_document_processing_error(self):
        """Test handling of document processing errors"""
        # Arrange
        file_path = "/path/to/invalid.pdf"
        self.mock_doc_processor.process_course_document.side_effect = Exception(
            "Invalid file format"
        )

        # Act
        course, chunk_count = self.rag_system.add_course_document(file_path)

        # Assert
        assert course is None
        assert chunk_count == 0

        # Verify vector store operations were not called
        self.mock_vector_store.add_course_metadata.assert_not_called()
        self.mock_vector_store.add_course_content.assert_not_called()

    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    def test_add_course_folder_success(self, mock_listdir, mock_exists):
        """Test successful course folder processing"""
        # Arrange
        folder_path = "/path/to/courses"
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "course1.pdf",
            "course2.docx",
            "readme.txt",
            "course3.txt",
        ]

        # Mock existing course titles
        self.mock_vector_store.get_existing_course_titles.return_value = []

        # Mock document processing
        test_course1 = Course(title="Course 1", lessons=[])
        test_course2 = Course(title="Course 2", lessons=[])
        test_course3 = Course(title="Course 3", lessons=[])

        test_chunks1 = [
            CourseChunk(content="Content 1", course_title="Course 1", chunk_index=0)
        ]
        test_chunks2 = [
            CourseChunk(content="Content 2", course_title="Course 2", chunk_index=0)
        ]
        test_chunks3 = [
            CourseChunk(content="Content 3", course_title="Course 3", chunk_index=0)
        ]

        self.mock_doc_processor.process_course_document.side_effect = [
            (test_course1, test_chunks1),
            (test_course2, test_chunks2),
            (test_course3, test_chunks3),
        ]

        # Act
        total_courses, total_chunks = self.rag_system.add_course_folder(folder_path)

        # Assert
        assert total_courses == 3
        assert total_chunks == 3

        # Verify all valid files were processed
        assert self.mock_doc_processor.process_course_document.call_count == 3

    @patch("rag_system.os.path.exists")
    def test_add_course_folder_nonexistent(self, mock_exists):
        """Test handling of nonexistent folder"""
        # Arrange
        folder_path = "/nonexistent/path"
        mock_exists.return_value = False

        # Act
        total_courses, total_chunks = self.rag_system.add_course_folder(folder_path)

        # Assert
        assert total_courses == 0
        assert total_chunks == 0

    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    def test_add_course_folder_with_existing_courses(self, mock_listdir, mock_exists):
        """Test course folder processing with existing courses"""
        # Arrange
        folder_path = "/path/to/courses"
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.pdf"]

        # Mock existing course titles - course1 already exists
        self.mock_vector_store.get_existing_course_titles.return_value = ["Course 1"]

        # Mock document processing
        test_course1 = Course(title="Course 1", lessons=[])
        test_course2 = Course(title="Course 2", lessons=[])

        test_chunks1 = [
            CourseChunk(content="Content 1", course_title="Course 1", chunk_index=0)
        ]
        test_chunks2 = [
            CourseChunk(content="Content 2", course_title="Course 2", chunk_index=0)
        ]

        self.mock_doc_processor.process_course_document.side_effect = [
            (test_course1, test_chunks1),
            (test_course2, test_chunks2),
        ]

        # Act
        total_courses, total_chunks = self.rag_system.add_course_folder(folder_path)

        # Assert
        assert total_courses == 1  # Only course2 was added (course1 already existed)
        assert total_chunks == 1

        # Verify only course2 was added to vector store
        self.mock_vector_store.add_course_metadata.assert_called_once_with(test_course2)
        self.mock_vector_store.add_course_content.assert_called_once_with(test_chunks2)

    def test_get_course_analytics(self):
        """Test course analytics retrieval"""
        # Arrange
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]

        # Act
        analytics = self.rag_system.get_course_analytics()

        # Assert
        expected = {
            "total_courses": 5,
            "course_titles": [
                "Course A",
                "Course B",
                "Course C",
                "Course D",
                "Course E",
            ],
        }
        assert analytics == expected

    def test_ai_generator_receives_correct_tools(self):
        """Test that AI generator receives properly configured tools"""
        # Arrange
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"},
        ]
        self.mock_tool_manager.get_tool_definitions.return_value = mock_tool_definitions

        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []

        # Act
        self.rag_system.query("Test query")

        # Assert
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["tools"] == mock_tool_definitions
        assert call_args[1]["tool_manager"] == self.mock_tool_manager

    def test_system_prompt_format(self):
        """Test that system prompt is properly formatted"""
        # Arrange
        query = "Test query"
        self.mock_ai_generator.generate_response.return_value = "Response"
        self.mock_tool_manager.get_last_sources.return_value = []

        # Act
        self.rag_system.query(query)

        # Assert
        call_args = self.mock_ai_generator.generate_response.call_args
        expected_prompt = f"Answer this question about course materials: {query}"
        assert call_args[1]["query"] == expected_prompt

    def test_conversation_flow_with_multiple_exchanges(self):
        """Test conversation flow with multiple exchanges"""
        # Arrange
        session_id = "test_session"

        # Mock conversation history that builds up
        self.mock_session_manager.get_conversation_history.side_effect = [
            None,  # First call - no history
            "User: First question\nAI: First response",  # Second call - has history
        ]

        self.mock_ai_generator.generate_response.side_effect = [
            "First response",
            "Second response building on first",
        ]

        self.mock_tool_manager.get_last_sources.return_value = []

        # Act
        response1, _ = self.rag_system.query("First question", session_id)
        response2, _ = self.rag_system.query("Follow-up question", session_id)

        # Assert
        assert response1 == "First response"
        assert response2 == "Second response building on first"

        # Verify conversation history was used in second call
        second_call_args = self.mock_ai_generator.generate_response.call_args_list[1]
        assert (
            second_call_args[1]["conversation_history"]
            == "User: First question\nAI: First response"
        )

        # Verify both exchanges were added to session
        assert self.mock_session_manager.add_exchange.call_count == 2


class TestRAGSystemErrorHandling:
    """Test error handling scenarios in RAGSystem"""

    def setup_method(self):
        """Set up test fixtures for error testing"""
        self.test_config = Config()
        self.test_config.ANTHROPIC_API_KEY = "test-key"

        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
            patch("rag_system.CourseSearchTool"),
            patch("rag_system.CourseOutlineTool"),
            patch("rag_system.ToolManager"),
        ):

            self.rag_system = RAGSystem(self.test_config)

    def test_query_ai_generator_exception(self):
        """Test handling of AI generator exceptions"""
        # Arrange
        self.rag_system.ai_generator = Mock()
        self.rag_system.ai_generator.generate_response.side_effect = Exception(
            "API error"
        )
        self.rag_system.tool_manager = Mock()
        self.rag_system.session_manager = Mock()

        # Act & Assert - should raise exception (not handled at RAG level)
        with pytest.raises(Exception, match="API error"):
            self.rag_system.query("Test query")

    def test_query_tool_manager_exception(self):
        """Test handling of tool manager exceptions"""
        # Arrange
        self.rag_system.ai_generator = Mock()
        self.rag_system.ai_generator.generate_response.return_value = "Response"
        self.rag_system.tool_manager = Mock()
        self.rag_system.tool_manager.get_last_sources.side_effect = Exception(
            "Tool error"
        )
        self.rag_system.session_manager = Mock()

        # Act & Assert - should raise exception
        with pytest.raises(Exception, match="Tool error"):
            self.rag_system.query("Test query")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
