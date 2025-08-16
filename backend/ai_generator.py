import anthropic
from typing import List, Optional, Dict, Any, Tuple


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Tool Usage Guidelines:
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews
- **Sequential Tool Access**: You can make up to 2 rounds of tool calls per query to gather comprehensive information
- **Search Strategy**: Start with broad searches, then narrow down based on initial results
- **Follow-up Searches**: Use initial search results to inform more targeted follow-up searches
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use content search tool first, then answer
- **Course outline/structure questions**: Use course outline tool to get complete course title, course link, and lesson details (lesson numbers and titles)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

For course outline queries, ensure your response includes:
- Complete course title
- Course link (if available)
- Complete list of lessons with numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initial messages list
        messages = [{"role": "user", "content": query}]

        # If no tools available, make direct call
        if not tools or not tool_manager:
            response = self._make_api_call(messages, system_content, tools=None)
            return response.content[0].text

        # Use recursive tool processing for multi-round capability
        return self._process_tool_rounds(
            messages, system_content, tools, tool_manager, max_rounds
        )

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _process_tool_rounds(
        self,
        messages: List[Dict],
        system_content: str,
        tools: Optional[List],
        tool_manager,
        max_rounds: int = 2,
        current_round: int = 1,
    ) -> str:
        """
        Process multiple rounds of tool calling recursively.

        Args:
            messages: Current conversation messages
            system_content: System prompt content
            tools: Available tools for this session
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds allowed (default 2)
            current_round: Current round number (1-based)

        Returns:
            Final response text after all rounds complete
        """
        # Termination condition: max rounds exceeded
        if current_round > max_rounds:
            # Final call without tools for clean response
            response = self._make_api_call(messages, system_content, tools=None)
            return response.content[0].text

        # Make API call with tools for this round
        response = self._make_api_call(messages, system_content, tools=tools)

        # Check if AI used tools
        if response.stop_reason != "tool_use":
            # AI provided direct response - we're done
            return response.content[0].text

        # Execute tools and update messages
        updated_messages, tool_success = self._execute_round_tools(
            response, messages, tool_manager
        )

        # Handle tool execution failure
        if not tool_success:
            # Make final call without tools
            response = self._make_api_call(updated_messages, system_content, tools=None)
            return response.content[0].text

        # Recursive call for next round
        return self._process_tool_rounds(
            updated_messages,
            system_content,
            tools,
            tool_manager,
            max_rounds,
            current_round + 1,
        )

    def _execute_round_tools(
        self, response, messages: List[Dict], tool_manager
    ) -> Tuple[List[Dict], bool]:
        """
        Execute all tool calls from a response and update message chain.

        Args:
            response: Claude response containing tool use requests
            messages: Current conversation messages
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (updated_messages, tool_success_flag)
        """
        # Create a copy to avoid modifying original
        updated_messages = messages.copy()

        # Add AI's tool use response to messages
        updated_messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls
        tool_results = []
        tool_success = True

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result,
                        }
                    )
                except Exception as e:
                    tool_success = False
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}",
                            "is_error": True,
                        }
                    )

        # Add tool results to messages
        if tool_results:
            updated_messages.append({"role": "user", "content": tool_results})

        return updated_messages, tool_success

    def _make_api_call(
        self, messages: List[Dict], system_content: str, tools: Optional[List] = None
    ):
        """
        Centralized API calling method with optional tool inclusion.

        Args:
            messages: Conversation messages for the API call
            system_content: System prompt content
            tools: Tools to include in the API call (None = no tools)

        Returns:
            Claude API response object
        """
        # Prepare API call parameters
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Make API call
        return self.client.messages.create(**api_params)
