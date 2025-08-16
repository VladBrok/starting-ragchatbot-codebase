import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator functionality"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        with patch('ai_generator.anthropic.Anthropic'):
            self.ai_generator = AIGenerator("test-api-key", "test-model")
            self.mock_client = Mock()
            self.ai_generator.client = self.mock_client

    def test_generate_response_without_tools(self):
        """Test generating response without tool usage"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Simple response without tools"
        mock_response.stop_reason = "end_turn"
        
        self.mock_client.messages.create.return_value = mock_response

        # Act
        response = self.ai_generator.generate_response("What is 2+2?")

        # Assert
        assert response == "Simple response without tools"
        
        # Verify API was called correctly
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["messages"][0]["content"] == "What is 2+2?"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert "tools" not in call_args[1]

    def test_generate_response_with_conversation_history(self):
        """Test generating response with conversation history"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history context"
        mock_response.stop_reason = "end_turn"
        
        self.mock_client.messages.create.return_value = mock_response
        
        history = "Previous conversation context"

        # Act
        response = self.ai_generator.generate_response(
            "Follow-up question", 
            conversation_history=history
        )

        # Assert
        assert response == "Response with history context"
        
        # Verify system prompt includes history
        call_args = self.mock_client.messages.create.call_args
        assert "Previous conversation context" in call_args[1]["system"]

    def test_generate_response_with_tools_no_tool_use(self):
        """Test generating response with tools available but not used"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response without using tools"
        mock_response.stop_reason = "end_turn"  # Not "tool_use"
        
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool", "description": "Test tool"}]
        mock_tool_manager = Mock()

        # Act
        response = self.ai_generator.generate_response(
            "General knowledge question",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert response == "Direct response without using tools"
        
        # Verify tools were provided to API
        call_args = self.mock_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"]["type"] == "auto"
        
        # Verify no tool execution occurred
        mock_tool_manager.execute_tool.assert_not_called()

    def test_generate_response_with_tool_use(self):
        """Test generating response when AI uses tools"""
        # Arrange
        # First response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "machine learning"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_block]
        initial_response.stop_reason = "tool_use"
        
        # Final response after tool execution
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Based on the search results, machine learning is..."
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: ML content found"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "What is machine learning?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert response == "Based on the search results, machine learning is..."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning"
        )
        
        # Verify two API calls were made
        assert self.mock_client.messages.create.call_count == 2

    def test_generate_response_tool_execution_error(self):
        """Test handling of tool execution errors"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "failing_tool"
        mock_tool_block.id = "tool_456"
        mock_tool_block.input = {"param": "value"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_block]
        initial_response.stop_reason = "tool_use"
        
        # Final response after tool error
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "I apologize, there was an error with the search."
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager that returns error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool 'failing_tool' not found"
        
        tools = [{"name": "failing_tool", "description": "Failing tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert response == "I apologize, there was an error with the search."
        mock_tool_manager.execute_tool.assert_called_once()

    def test_generate_response_multiple_tool_calls(self):
        """Test handling multiple tool calls in single response"""
        # Arrange
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "tool_1"
        mock_tool_block1.id = "tool_1_id"
        mock_tool_block1.input = {"param": "value1"}
        
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "tool_2"
        mock_tool_block2.id = "tool_2_id"
        mock_tool_block2.input = {"param": "value2"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_block1, mock_tool_block2]
        initial_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Combined results from both tools"
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        tools = [{"name": "tool_1"}, {"name": "tool_2"}]

        # Act
        response = self.ai_generator.generate_response(
            "Multi-tool query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert response == "Combined results from both tools"
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify both tools were called with correct parameters
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("tool_1",)
        assert calls[0][1] == {"param": "value1"}
        assert calls[1][0] == ("tool_2",)
        assert calls[1][1] == {"param": "value2"}

    def test_handle_tool_execution_message_structure(self):
        """Test that tool execution creates proper message structure"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "test_tool"
        mock_tool_block.id = "tool_id_123"
        mock_tool_block.input = {"query": "test"}
        
        initial_response = Mock()
        initial_response.content = [mock_tool_block]
        initial_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer"
        
        self.mock_client.messages.create.return_value = final_response
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        base_params = {
            "messages": [{"role": "user", "content": "Test question"}],
            "system": "Test system prompt"
        }

        # Act
        result = self.ai_generator._handle_tool_execution(
            initial_response, 
            base_params, 
            mock_tool_manager
        )

        # Assert
        assert result == "Final answer"
        
        # Verify final API call had correct message structure
        final_call_args = self.mock_client.messages.create.call_args
        messages = final_call_args[1]["messages"]
        
        # Should have: user message, assistant message with tool use, user message with tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_id_123"
        assert messages[2]["content"][0]["content"] == "Tool result"

    def test_system_prompt_structure(self):
        """Test that system prompt contains expected instructions"""
        # Assert key components are in system prompt
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        assert "Tool Usage Guidelines" in system_prompt
        assert "Content Search Tool" in system_prompt
        assert "Course Outline Tool" in system_prompt
        assert "Sequential Tool Access" in system_prompt
        assert "Response Protocol" in system_prompt
        assert "No meta-commentary" in system_prompt

    def test_base_params_structure(self):
        """Test that base API parameters are properly structured"""
        # Arrange & Act
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Assert
        assert ai_gen.base_params["model"] == "test-model"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800

    def test_generate_response_no_tool_manager_with_tool_use(self):
        """Test behavior when tools are used but no tool_manager provided"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "test_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "test"}
        
        # When no tool_manager is provided, the call should be made without tools
        # so we don't expect tool_use in the response
        direct_response = Mock()
        direct_response.content = [Mock()]
        direct_response.content[0].text = "Direct response without tools"
        
        self.mock_client.messages.create.return_value = direct_response
        
        tools = [{"name": "test_tool", "description": "Test tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Query that would need tools",
            tools=tools,
            tool_manager=None  # No tool manager provided
        )

        # Assert - should return direct response when no tool_manager is provided
        assert response == "Direct response without tools"

    def test_generate_response_with_non_tool_content(self):
        """Test handling response with mixed content types"""
        # Arrange
        mock_text_block = Mock()
        mock_text_block.type = "text"
        mock_text_block.text = "Here's some explanation"
        
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "search"}
        
        initial_response = Mock()
        initial_response.content = [mock_text_block, mock_tool_block]
        initial_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final response"
        
        self.mock_client.messages.create.side_effect = [initial_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        tools = [{"name": "search_tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Query",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert response == "Final response"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_tool", query="search"
        )

    def test_api_parameters_isolation(self):
        """Test that API parameters don't interfere between calls"""
        # Arrange
        mock_response1 = Mock()
        mock_response1.content = [Mock()]
        mock_response1.content[0].text = "First response"
        mock_response1.stop_reason = "end_turn"
        
        mock_response2 = Mock()
        mock_response2.content = [Mock()]
        mock_response2.content[0].text = "Second response"
        mock_response2.stop_reason = "end_turn"
        
        self.mock_client.messages.create.side_effect = [mock_response1, mock_response2]

        # Act - Make two separate calls
        response1 = self.ai_generator.generate_response("First query", conversation_history="History 1")
        response2 = self.ai_generator.generate_response("Second query", conversation_history="History 2")

        # Assert
        assert response1 == "First response"
        assert response2 == "Second response"
        
        # Verify each call had correct isolated parameters
        calls = self.mock_client.messages.create.call_args_list
        assert "History 1" in calls[0][1]["system"]
        assert "History 2" in calls[1][1]["system"]
        assert "History 1" not in calls[1][1]["system"]
        assert "History 2" not in calls[0][1]["system"]


class TestSequentialToolCalling:
    """Test suite for sequential tool calling functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        with patch('ai_generator.anthropic.Anthropic'):
            self.ai_generator = AIGenerator("test-api-key", "test-model")
            self.mock_client = Mock()
            self.ai_generator.client = self.mock_client

    def test_single_round_completion(self):
        """Test that single round works correctly and terminates"""
        # Arrange
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response without tools"
        mock_response.stop_reason = "end_turn"  # No tool use
        
        self.mock_client.messages.create.return_value = mock_response
        
        tools = [{"name": "test_tool", "description": "Test tool"}]
        mock_tool_manager = Mock()

        # Act
        response = self.ai_generator.generate_response(
            "Simple question",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Assert
        assert response == "Direct response without tools"
        
        # Verify only one API call was made (with tools)
        assert self.mock_client.messages.create.call_count == 1
        call_args = self.mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        
        # Verify no tool execution occurred
        mock_tool_manager.execute_tool.assert_not_called()

    def test_two_round_completion(self):
        """Test successful two-round tool calling"""
        # Arrange
        # Round 1: AI uses tools
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_tool"
        mock_tool_block1.id = "tool_1_id"
        mock_tool_block1.input = {"query": "initial search"}
        
        round1_response = Mock()
        round1_response.content = [mock_tool_block1]
        round1_response.stop_reason = "tool_use"
        
        # Round 2: AI uses tools again
        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_tool"
        mock_tool_block2.id = "tool_2_id"
        mock_tool_block2.input = {"query": "follow-up search"}
        
        round2_response = Mock()
        round2_response.content = [mock_tool_block2]
        round2_response.stop_reason = "tool_use"
        
        # Final response after max rounds
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer after 2 rounds"
        
        self.mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        tools = [{"name": "search_tool", "description": "Search tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Complex question requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Assert
        assert response == "Final answer after 2 rounds"
        
        # Verify 3 API calls were made (round1, round2, final)
        assert self.mock_client.messages.create.call_count == 3
        
        # Verify 2 tool executions occurred
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tools were called with correct parameters
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_tool",)
        assert calls[0][1] == {"query": "initial search"}
        assert calls[1][0] == ("search_tool",)
        assert calls[1][1] == {"query": "follow-up search"}

    def test_early_termination_no_tools_round_two(self):
        """Test termination when no tools used in second round"""
        # Arrange
        # Round 1: AI uses tools
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "search"}
        
        round1_response = Mock()
        round1_response.content = [mock_tool_block]
        round1_response.stop_reason = "tool_use"
        
        # Round 2: AI provides direct response (no tools)
        round2_response = Mock()
        round2_response.content = [Mock()]
        round2_response.content[0].text = "Complete answer after 1 tool round"
        round2_response.stop_reason = "end_turn"
        
        self.mock_client.messages.create.side_effect = [round1_response, round2_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        tools = [{"name": "search_tool", "description": "Search tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Question that needs one search",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Assert
        assert response == "Complete answer after 1 tool round"
        
        # Verify 2 API calls were made (round1, round2 with direct response)
        assert self.mock_client.messages.create.call_count == 2
        
        # Verify 1 tool execution occurred
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_max_rounds_enforcement(self):
        """Test that system enforces max_rounds limit"""
        # Arrange - AI always tries to use tools
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "search"}
        
        tool_use_response = Mock()
        tool_use_response.content = [mock_tool_block]
        tool_use_response.stop_reason = "tool_use"
        
        # Final response after max rounds reached
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer (max rounds reached)"
        
        # Return tool_use for first 2 calls, then final response
        self.mock_client.messages.create.side_effect = [tool_use_response, tool_use_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        tools = [{"name": "search_tool", "description": "Search tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Question that AI wants to search extensively",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Assert
        assert response == "Final answer (max rounds reached)"
        
        # Verify exactly 3 API calls (2 tool rounds + 1 final without tools)
        assert self.mock_client.messages.create.call_count == 3
        
        # Verify exactly 2 tool executions (max_rounds)
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_tool_execution_error_handling(self):
        """Test graceful handling of tool execution failures"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "failing_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "test"}
        
        tool_use_response = Mock()
        tool_use_response.content = [mock_tool_block]
        tool_use_response.stop_reason = "tool_use"
        
        # Final response after tool error
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Answer despite tool error"
        
        self.mock_client.messages.create.side_effect = [tool_use_response, final_response]
        
        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        tools = [{"name": "failing_tool", "description": "Failing tool"}]

        # Act
        response = self.ai_generator.generate_response(
            "Question with failing tool",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Assert
        assert response == "Answer despite tool error"
        
        # Verify 2 API calls (tool round + final without tools due to error)
        assert self.mock_client.messages.create.call_count == 2
        
        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once()

    def test_conversation_context_preservation(self):
        """Test that conversation context is maintained across rounds"""
        # Arrange
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "search"}
        
        tool_use_response = Mock()
        tool_use_response.content = [mock_tool_block]
        tool_use_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer"
        
        self.mock_client.messages.create.side_effect = [tool_use_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        tools = [{"name": "search_tool", "description": "Search tool"}]
        conversation_history = "Previous conversation context"

        # Act
        response = self.ai_generator.generate_response(
            "Follow-up question",
            conversation_history=conversation_history,
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Assert
        assert response == "Final answer"
        
        # Verify conversation history is included in both API calls
        calls = self.mock_client.messages.create.call_args_list
        for call in calls:
            assert "Previous conversation context" in call[1]["system"]

    def test_max_rounds_parameter_customization(self):
        """Test that max_rounds parameter can be customized"""
        # Arrange - AI always tries to use tools
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_tool"
        mock_tool_block.id = "tool_id"
        mock_tool_block.input = {"query": "search"}
        
        tool_use_response = Mock()
        tool_use_response.content = [mock_tool_block]
        tool_use_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer (1 round max)"
        
        # Should get tool_use once, then final response
        self.mock_client.messages.create.side_effect = [tool_use_response, final_response]
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search result"
        
        tools = [{"name": "search_tool", "description": "Search tool"}]

        # Act - Set max_rounds to 1
        response = self.ai_generator.generate_response(
            "Question with max_rounds=1",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=1
        )

        # Assert
        assert response == "Final answer (1 round max)"
        
        # Verify exactly 2 API calls (1 tool round + 1 final)
        assert self.mock_client.messages.create.call_count == 2
        
        # Verify exactly 1 tool execution
        assert mock_tool_manager.execute_tool.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])