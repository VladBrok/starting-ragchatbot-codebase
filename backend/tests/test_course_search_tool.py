import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool execute method"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_vector_store = Mock()
        self.tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search(self):
        """Test successful search with valid query and results"""
        # Arrange
        mock_results = SearchResults(
            documents=["Course content about machine learning", "More ML concepts"],
            metadata=[
                {"course_title": "Introduction to ML", "lesson_number": 1},
                {"course_title": "Introduction to ML", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "http://example.com/lesson1"
        )

        # Act
        result = self.tool.execute("machine learning")

        # Assert
        assert result is not None
        assert "Introduction to ML" in result
        assert "machine learning" in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

    def test_execute_with_course_name_filter(self):
        """Test search with course name filtering"""
        # Arrange
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 1}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.tool.execute("query", course_name="Specific Course")

        # Assert
        assert "Specific Course" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="query", course_name="Specific Course", lesson_number=None
        )

    def test_execute_with_lesson_number_filter(self):
        """Test search with lesson number filtering"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Some Course", "lesson_number": 3}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "http://example.com/lesson3"
        )

        # Act
        result = self.tool.execute("query", lesson_number=3)

        # Assert
        assert "Lesson 3" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="query", course_name=None, lesson_number=3
        )

    def test_execute_with_both_filters(self):
        """Test search with both course name and lesson number filtering"""
        # Arrange
        mock_results = SearchResults(
            documents=["Specific filtered content"],
            metadata=[{"course_title": "Target Course", "lesson_number": 5}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = (
            "http://example.com/target-lesson5"
        )

        # Act
        result = self.tool.execute(
            "query", course_name="Target Course", lesson_number=5
        )

        # Assert
        assert "Target Course" in result
        assert "Lesson 5" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="query", course_name="Target Course", lesson_number=5
        )

    def test_execute_empty_results(self):
        """Test handling of empty search results"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute("nonexistent query")

        # Assert
        assert "No relevant content found" in result
        assert result == "No relevant content found."

    def test_execute_empty_results_with_course_filter(self):
        """Test empty results with course filtering includes course name in message"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute("query", course_name="Missing Course")

        # Assert
        assert "No relevant content found in course 'Missing Course'." in result

    def test_execute_empty_results_with_lesson_filter(self):
        """Test empty results with lesson filtering includes lesson number in message"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute("query", lesson_number=99)

        # Assert
        assert "No relevant content found in lesson 99." in result

    def test_execute_empty_results_with_both_filters(self):
        """Test empty results with both filters includes both in message"""
        # Arrange
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute(
            "query", course_name="Missing Course", lesson_number=99
        )

        # Assert
        assert (
            "No relevant content found in course 'Missing Course' in lesson 99."
            in result
        )

    def test_execute_search_error(self):
        """Test handling of search errors"""
        # Arrange
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute("query")

        # Assert
        assert result == "Database connection failed"

    def test_execute_max_results_zero_issue(self):
        """Test if MAX_RESULTS=0 configuration causes issues"""
        # Arrange - simulate the MAX_RESULTS=0 scenario
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute("valid query")

        # Assert - this should return "no content found" not fail
        assert "No relevant content found" in result
        # This test will help identify if the issue is MAX_RESULTS=0

    def test_format_results_with_sources_tracking(self):
        """Test that sources are properly tracked for UI"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.side_effect = [
            "http://example.com/courseA-lesson1",
            "http://example.com/courseB-lesson2",
        ]

        # Act
        result = self.tool.execute("query")

        # Assert
        assert len(self.tool.last_sources) == 2
        assert self.tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert self.tool.last_sources[0]["link"] == "http://example.com/courseA-lesson1"
        assert self.tool.last_sources[1]["text"] == "Course B - Lesson 2"
        assert self.tool.last_sources[1]["link"] == "http://example.com/courseB-lesson2"

    def test_format_results_without_lesson_numbers(self):
        """Test formatting when lesson numbers are missing"""
        # Arrange
        mock_results = SearchResults(
            documents=["General content"],
            metadata=[{"course_title": "General Course", "lesson_number": None}],
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.tool.execute("query")

        # Assert
        assert "[General Course]" in result
        assert "Lesson" not in result  # Should not include lesson info
        assert len(self.tool.last_sources) == 1
        assert self.tool.last_sources[0]["text"] == "General Course"
        assert self.tool.last_sources[0]["link"] is None

    def test_get_tool_definition(self):
        """Test that tool definition is properly structured for Anthropic"""
        # Act
        definition = self.tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_with_unknown_metadata_keys(self):
        """Test handling of unexpected metadata structure"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"unexpected_key": "value"}],  # Missing expected keys
            distances=[0.1],
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.tool.execute("query")

        # Assert - should handle gracefully with default values
        assert "[unknown]" in result  # Default course title
        assert len(self.tool.last_sources) == 1
        assert self.tool.last_sources[0]["text"] == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
