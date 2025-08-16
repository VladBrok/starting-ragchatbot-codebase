import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Test suite for VectorStore functionality"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()

        # Mock ChromaDB components to avoid actual database operations
        with patch("vector_store.chromadb.PersistentClient") as mock_client:
            with patch(
                "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding:
                self.mock_client_instance = Mock()
                self.mock_course_catalog = Mock()
                self.mock_course_content = Mock()

                mock_client.return_value = self.mock_client_instance
                self.mock_client_instance.get_or_create_collection.side_effect = [
                    self.mock_course_catalog,
                    self.mock_course_content,
                ]

                # Create VectorStore with mocked dependencies
                self.vector_store = VectorStore(
                    self.temp_dir, "test-model", max_results=5
                )
                self.vector_store.course_catalog = self.mock_course_catalog
                self.vector_store.course_content = self.mock_course_content

    def teardown_method(self):
        """Clean up after each test method"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_search_successful_query(self):
        """Test successful search with valid query"""
        # Arrange
        mock_chroma_results = {
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [
                [
                    {"course_title": "Test Course", "lesson_number": 1},
                    {"course_title": "Test Course", "lesson_number": 2},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
        self.mock_course_content.query.return_value = mock_chroma_results

        # Act
        results = self.vector_store.search("test query")

        # Assert
        assert not results.error
        assert len(results.documents) == 2
        assert results.documents[0] == "Document 1"
        assert results.metadata[0]["course_title"] == "Test Course"
        assert results.distances[0] == 0.1

        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"], n_results=5, where=None
        )

    def test_search_with_max_results_zero(self):
        """Test search behavior when max_results is 0 (critical issue)"""
        # Arrange - simulate the MAX_RESULTS=0 configuration issue
        self.vector_store.max_results = 0
        mock_chroma_results = {
            "documents": [[]],  # Empty results due to n_results=0
            "metadatas": [[]],
            "distances": [[]],
        }
        self.mock_course_content.query.return_value = mock_chroma_results

        # Act
        results = self.vector_store.search("test query")

        # Assert
        assert not results.error  # Should not error
        assert results.is_empty()  # But should be empty

        self.mock_course_content.query.assert_called_once_with(
            query_texts=["test query"], n_results=0, where=None  # This is the problem!
        )

    def test_search_with_course_name_resolution(self):
        """Test search with course name that requires resolution"""
        # Arrange
        # Mock course name resolution
        mock_catalog_results = {
            "documents": [["Introduction to Machine Learning"]],
            "metadatas": [[{"title": "Introduction to Machine Learning"}]],
        }
        self.mock_course_catalog.query.return_value = mock_catalog_results

        # Mock content search results
        mock_content_results = {
            "documents": [["ML content"]],
            "metadatas": [
                [
                    {
                        "course_title": "Introduction to Machine Learning",
                        "lesson_number": 1,
                    }
                ]
            ],
            "distances": [[0.1]],
        }
        self.mock_course_content.query.return_value = mock_content_results

        # Act
        results = self.vector_store.search("concepts", course_name="ML")

        # Assert
        assert not results.error
        assert len(results.documents) == 1

        # Should first resolve course name
        self.mock_course_catalog.query.assert_called_once_with(
            query_texts=["ML"], n_results=1
        )

        # Then search content with resolved course name
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["concepts"],
            n_results=5,
            where={"course_title": "Introduction to Machine Learning"},
        )

    def test_search_course_name_not_found(self):
        """Test search when course name cannot be resolved"""
        # Arrange
        mock_catalog_results = {
            "documents": [[]],  # No matching course
            "metadatas": [[]],
        }
        self.mock_course_catalog.query.return_value = mock_catalog_results

        # Act
        results = self.vector_store.search("query", course_name="Nonexistent Course")

        # Assert
        assert results.error is not None
        assert "No course found matching 'Nonexistent Course'" in results.error
        assert results.is_empty()

    def test_search_with_lesson_number_filter(self):
        """Test search with lesson number filtering"""
        # Arrange
        mock_content_results = {
            "documents": [["Lesson 3 content"]],
            "metadatas": [[{"course_title": "Some Course", "lesson_number": 3}]],
            "distances": [[0.1]],
        }
        self.mock_course_content.query.return_value = mock_content_results

        # Act
        results = self.vector_store.search("query", lesson_number=3)

        # Assert
        assert not results.error
        assert len(results.documents) == 1

        self.mock_course_content.query.assert_called_once_with(
            query_texts=["query"], n_results=5, where={"lesson_number": 3}
        )

    def test_search_with_both_course_and_lesson_filters(self):
        """Test search with both course name and lesson number filters"""
        # Arrange
        # Mock course name resolution
        mock_catalog_results = {
            "documents": [["Target Course"]],
            "metadatas": [[{"title": "Target Course"}]],
        }
        self.mock_course_catalog.query.return_value = mock_catalog_results

        # Mock content search
        mock_content_results = {
            "documents": [["Specific lesson content"]],
            "metadatas": [[{"course_title": "Target Course", "lesson_number": 5}]],
            "distances": [[0.1]],
        }
        self.mock_course_content.query.return_value = mock_content_results

        # Act
        results = self.vector_store.search(
            "query", course_name="Target", lesson_number=5
        )

        # Assert
        assert not results.error

        # Should build complex filter
        expected_filter = {
            "$and": [{"course_title": "Target Course"}, {"lesson_number": 5}]
        }

        self.mock_course_content.query.assert_called_once_with(
            query_texts=["query"], n_results=5, where=expected_filter
        )

    def test_search_with_custom_limit(self):
        """Test search with custom result limit"""
        # Arrange
        mock_content_results = {
            "documents": [["Doc 1", "Doc 2", "Doc 3"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        self.mock_course_content.query.return_value = mock_content_results

        # Act
        results = self.vector_store.search("query", limit=10)

        # Assert
        self.mock_course_content.query.assert_called_once_with(
            query_texts=["query"],
            n_results=10,  # Should use custom limit, not default max_results
            where=None,
        )

    def test_search_chroma_exception(self):
        """Test handling of ChromaDB exceptions during search"""
        # Arrange
        self.mock_course_content.query.side_effect = Exception("Database error")

        # Act
        results = self.vector_store.search("query")

        # Assert
        assert results.error is not None
        assert "Search error: Database error" in results.error
        assert results.is_empty()

    def test_resolve_course_name_success(self):
        """Test successful course name resolution"""
        # Arrange
        mock_catalog_results = {
            "documents": [["Machine Learning Fundamentals"]],
            "metadatas": [[{"title": "Machine Learning Fundamentals"}]],
        }
        self.mock_course_catalog.query.return_value = mock_catalog_results

        # Act
        resolved_name = self.vector_store._resolve_course_name("ML Fund")

        # Assert
        assert resolved_name == "Machine Learning Fundamentals"

    def test_resolve_course_name_no_match(self):
        """Test course name resolution when no match found"""
        # Arrange
        mock_catalog_results = {"documents": [[]], "metadatas": [[]]}
        self.mock_course_catalog.query.return_value = mock_catalog_results

        # Act
        resolved_name = self.vector_store._resolve_course_name("Nonexistent")

        # Assert
        assert resolved_name is None

    def test_resolve_course_name_exception(self):
        """Test course name resolution with exception"""
        # Arrange
        self.mock_course_catalog.query.side_effect = Exception("Catalog error")

        # Act
        resolved_name = self.vector_store._resolve_course_name("Course")

        # Assert
        assert resolved_name is None

    def test_build_filter_no_filters(self):
        """Test filter building with no parameters"""
        # Act
        filter_dict = self.vector_store._build_filter(None, None)

        # Assert
        assert filter_dict is None

    def test_build_filter_course_only(self):
        """Test filter building with course title only"""
        # Act
        filter_dict = self.vector_store._build_filter("Test Course", None)

        # Assert
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self):
        """Test filter building with lesson number only"""
        # Act
        filter_dict = self.vector_store._build_filter(None, 3)

        # Assert
        assert filter_dict == {"lesson_number": 3}

    def test_build_filter_both_parameters(self):
        """Test filter building with both course and lesson"""
        # Act
        filter_dict = self.vector_store._build_filter("Test Course", 3)

        # Assert
        expected = {"$and": [{"course_title": "Test Course"}, {"lesson_number": 3}]}
        assert filter_dict == expected

    def test_add_course_metadata(self):
        """Test adding course metadata to catalog"""
        # Arrange
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://example.com/course",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Lesson 1",
                    lesson_link="http://example.com/lesson1",
                ),
                Lesson(
                    lesson_number=2,
                    title="Lesson 2",
                    lesson_link="http://example.com/lesson2",
                ),
            ],
        )

        # Act
        self.vector_store.add_course_metadata(course)

        # Assert
        self.mock_course_catalog.add.assert_called_once()
        call_args = self.mock_course_catalog.add.call_args

        assert call_args[1]["documents"] == ["Test Course"]
        assert call_args[1]["ids"] == ["Test Course"]

        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Test Course"
        assert metadata["instructor"] == "Test Instructor"
        assert metadata["course_link"] == "http://example.com/course"
        assert metadata["lesson_count"] == 2
        assert "lessons_json" in metadata

    def test_add_course_content(self):
        """Test adding course content chunks"""
        # Arrange
        chunks = [
            CourseChunk(
                content="Content 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
            ),
            CourseChunk(
                content="Content 2",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
            ),
        ]

        # Act
        self.vector_store.add_course_content(chunks)

        # Assert
        self.mock_course_content.add.assert_called_once()
        call_args = self.mock_course_content.add.call_args

        assert call_args[1]["documents"] == ["Content 1", "Content 2"]
        assert call_args[1]["ids"] == ["Test_Course_0", "Test_Course_1"]
        assert len(call_args[1]["metadatas"]) == 2

    def test_get_existing_course_titles(self):
        """Test retrieving existing course titles"""
        # Arrange
        mock_get_results = {"ids": ["Course 1", "Course 2", "Course 3"]}
        self.mock_course_catalog.get.return_value = mock_get_results

        # Act
        titles = self.vector_store.get_existing_course_titles()

        # Assert
        assert titles == ["Course 1", "Course 2", "Course 3"]

    def test_get_existing_course_titles_empty(self):
        """Test retrieving course titles when database is empty"""
        # Arrange
        mock_get_results = {"ids": []}
        self.mock_course_catalog.get.return_value = mock_get_results

        # Act
        titles = self.vector_store.get_existing_course_titles()

        # Assert
        assert titles == []

    def test_get_existing_course_titles_exception(self):
        """Test retrieving course titles with exception"""
        # Arrange
        self.mock_course_catalog.get.side_effect = Exception("Database error")

        # Act
        titles = self.vector_store.get_existing_course_titles()

        # Assert
        assert titles == []

    def test_get_course_count(self):
        """Test getting course count"""
        # Arrange
        mock_get_results = {"ids": ["Course 1", "Course 2"]}
        self.mock_course_catalog.get.return_value = mock_get_results

        # Act
        count = self.vector_store.get_course_count()

        # Assert
        assert count == 2

    def test_get_lesson_link(self):
        """Test retrieving lesson link"""
        # Arrange
        lessons_json = '[{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://example.com/lesson1"}]'
        mock_get_results = {"metadatas": [{"lessons_json": lessons_json}]}
        self.mock_course_catalog.get.return_value = mock_get_results

        # Act
        link = self.vector_store.get_lesson_link("Test Course", 1)

        # Assert
        assert link == "http://example.com/lesson1"
        self.mock_course_catalog.get.assert_called_once_with(ids=["Test Course"])

    def test_get_lesson_link_not_found(self):
        """Test retrieving lesson link when lesson not found"""
        # Arrange
        lessons_json = '[{"lesson_number": 2, "lesson_title": "Other", "lesson_link": "http://example.com/lesson2"}]'
        mock_get_results = {"metadatas": [{"lessons_json": lessons_json}]}
        self.mock_course_catalog.get.return_value = mock_get_results

        # Act
        link = self.vector_store.get_lesson_link(
            "Test Course", 1
        )  # Looking for lesson 1, but only lesson 2 exists

        # Assert
        assert link is None


class TestSearchResults:
    """Test suite for SearchResults class"""

    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results"""
        # Arrange
        chroma_results = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        # Act
        results = SearchResults.from_chroma(chroma_results)

        # Assert
        assert results.documents == ["Doc 1", "Doc 2"]
        assert results.metadata == [{"key": "value1"}, {"key": "value2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        # Arrange
        chroma_results = {"documents": [], "metadatas": [], "distances": []}

        # Act
        results = SearchResults.from_chroma(chroma_results)

        # Assert
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        # Act
        results = SearchResults.empty("Test error message")

        # Assert
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error message"
        assert results.is_empty()

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        # Arrange
        results = SearchResults(documents=[], metadata=[], distances=[])

        # Act & Assert
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        # Arrange
        results = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])

        # Act & Assert
        assert results.is_empty() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
