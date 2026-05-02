"""
AI Agent based on LangGraph with RouterAPI (OpenAI Compatible), Tavily search, and math tools.
"""

import os
import logging
import base64
from typing import TypedDict, Annotated, Sequence, Optional, List, Union
from operator import add

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from tools.math_tool import AddNumbersTool

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add]
    is_final: bool
    memory_limit: int


class AIAgent:
    """AI Agent using LangGraph with RouterAPI and Tavily search."""
    
    def __init__(
        self,
        routerapi_base_url: str = None,
        routerapi_api_key: str = None,
        tavily_api_key: str = None,
        model_name: str = None,
        tavily_max_results: int = None,
        memory_limit: int = None
    ):
        """
        Initialize the agent.
        
        Args:
            routerapi_base_url: Base URL for RouterAPI (OpenAI Compatible)
            routerapi_api_key: API key for RouterAPI
            tavily_api_key: API key for Tavily search
            model_name: Model name to use
            tavily_max_results: Maximum number of results from Tavily search
            memory_limit: Maximum number of messages to keep in memory (short memory)
        """
        logger.debug("AIAgent: Initializing agent")
        logger.debug(f"AIAgent: Parameters - model_name={model_name}, memory_limit={memory_limit}")
        
        # Get from environment if not provided
        self.routerapi_base_url = routerapi_base_url or os.getenv("ROUTERAPI_BASE_URL")
        self.routerapi_api_key = routerapi_api_key or os.getenv("ROUTERAPI_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.model_name = model_name or os.getenv("MODEL_NAME", "qwen/qwen3.6-flash")
        self.tavily_max_results = tavily_max_results or int(os.getenv("TAVILY_MAX_RESULTS", "5"))
        self.memory_limit = memory_limit or int(os.getenv("MEMORY_LIMIT", "10"))
        
        logger.debug(f"AIAgent: Using RouterAPI URL: {self.routerapi_base_url}")
        logger.debug(f"AIAgent: Using model: {self.model_name}")
        logger.debug(f"AIAgent: Tavily max results: {self.tavily_max_results}")
        logger.debug(f"AIAgent: Memory limit: {self.memory_limit}")
        
        if not all([self.routerapi_base_url, self.routerapi_api_key, self.tavily_api_key]):
            logger.error("AIAgent: Missing required environment variables")
            raise ValueError(
                "Missing required environment variables. "
                "Please set ROUTERAPI_BASE_URL, ROUTERAPI_API_KEY, and TAVILY_API_KEY."
            )
        
        # Initialize LLM with RouterAPI (OpenAI Compatible)
        # RouterAPI typically uses https://openrouter.ai/api/v1 as base URL
        logger.debug("AIAgent: Creating ChatOpenAI instance")
        self.llm = ChatOpenAI(
            base_url=self.routerapi_base_url,
            api_key=self.routerapi_api_key,
            model=self.model_name,
            temperature=0.7,
            timeout=300.0,
            max_retries=2,
            # Disable proxy to avoid SSL/TLS issues with local proxies
            http_client=None
        )
        logger.debug("AIAgent: ChatOpenAI instance created")
        
        # Initialize Tavily search tool
        logger.debug("AIAgent: Creating TavilySearch instance")
        self.search_tool = TavilySearch(
            api_key=self.tavily_api_key,
            max_results=self.tavily_max_results
        )
        logger.debug("AIAgent: TavilySearch instance created")
        
        # Initialize math tool for adding numbers
        logger.debug("AIAgent: Creating AddNumbersTool instance")
        self.math_tool = AddNumbersTool()
        logger.debug("AIAgent: AddNumbersTool instance created")
        
        # Define tools
        self.tools = [self.search_tool, self.math_tool]
        
        # Build the graph
        logger.debug("AIAgent: Building graph")
        self.graph = self._build_graph()
        logger.debug("AIAgent: Graph built successfully")
    
    def _add_messages(self, state: AgentState, new_messages: Sequence[BaseMessage]) -> AgentState:
        """Add new messages to the state."""
        logger.debug(f"_add_messages: Adding {len(new_messages)} messages to state")
        return {"messages": new_messages}
    
    def _call_model(self, state: AgentState) -> AgentState:
        """Call the LLM to generate a response."""
        messages = state["messages"]
        
        logger.debug(f"_call_model: Starting with {len(messages)} messages")
        
        # Log the full context being sent to the model
        logger.info("=" * 80)
        logger.info("AI MODEL CALL - Full Context:")
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                logger.info(f"[Human {i+1}]: {msg.content}")
            elif isinstance(msg, AIMessage):
                logger.info(f"[AI {i+1}]: {msg.content}")
                if msg.tool_calls:
                    logger.info(f"[AI {i+1} Tool Calls]: {msg.tool_calls}")
            elif isinstance(msg, SystemMessage):
                logger.info(f"[System]: {msg.content}")
            else:
                logger.info(f"Ignored Message type")
        logger.info("=" * 80)
        
        # Bind tools to the LLM
        logger.debug(f"_call_model: Binding {len(self.tools)} tools to LLM")
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Get response from LLM
        logger.debug("_call_model: Invoking LLM with tools")
        response = llm_with_tools.invoke(messages)
        logger.debug(f"_call_model: LLM returned response with content length: {len(response.content) if response.content else 0}")
        
        # Log the full response from the model
        logger.info("AI MODEL RESPONSE:")
        logger.info(f"[AI Response Content]: {response.content}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"[AI Response Tool Calls]: {response.tool_calls}")
        logger.info("=" * 80)
        
        return {"messages": [response]}
    
    def _route_message(self, state: AgentState) -> str:
        """Determine if we need to use a tool or if we're done."""
        messages = state["messages"]
        last_message = messages[-1]
        
        logger.debug(f"_route_message: Checking last message type: {type(last_message).__name__}")
        
        # If the last message has tool calls, we need to use tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_call_count = len(last_message.tool_calls)
            logger.debug(f"_route_message: Found {tool_call_count} tool calls, routing to 'tools'")
            return "tools"
        
        logger.debug("_route_message: No tool calls found, routing to 'end'")
        # Otherwise, we're done
        return "end"
    
    def _run_tools(self, state: AgentState) -> AgentState:
        """Run the tools based on tool calls in the last message."""
        messages = state["messages"]
        last_message = messages[-1]
        
        logger.debug(f"_run_tools: Starting with {len(messages)} messages")
        
        # Execute tool calls
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"].lower().replace(" ", "_")
            tool_args = tool_call["args"]
            
            logger.debug(f"_run_tools: Executing tool '{tool_name}' with args: {tool_args}")
            
            # Log the search query
            logger.info("=" * 80)
            logger.info(f"SEARCH TOOL CALL - Tool: {tool_name}")
            logger.info(f"Search Query/Args: {tool_args}")
            
            # Find and execute the tool
            selected_tool = None
            for tool in self.tools:
                if tool.name.lower() == tool_name:
                    selected_tool = tool
                    break
            
            if selected_tool:
                logger.debug(f"_run_tools: Found tool '{tool_name}', invoking...")
                tool_response = selected_tool.invoke(tool_args)
                logger.debug(f"_run_tools: Tool '{tool_name}' returned response: {type(tool_response).__name__}")
                
                # Log the search response
                logger.info(f"Search Response: {tool_response}")
                logger.info("=" * 80)
                
                # Convert tool response to a proper message
                # Tavily returns a dict with results, query, etc.
                if isinstance(tool_response, dict):
                    # Format the tool response as a string message
                    response_text = self._format_tool_response(tool_response)
                    tool_messages.append(AIMessage(content=f"[Tool: {tool_name}] {response_text}"))
                elif isinstance(tool_response, (int, float)):
                    # Handle math tool response (numeric result)
                    response_text = self._format_math_response(tool_response)
                    tool_messages.append(AIMessage(content=f"[Tool: {tool_name}] {response_text}"))
                else:
                    tool_messages.append(tool_response)
            else:
                logger.warning(f"_run_tools: Tool '{tool_name}' not found in available tools: {[t.name for t in self.tools]}")
        
        logger.debug(f"_run_tools: Returning {len(tool_messages)} tool messages")
        # Return messages with tool results
        return {"messages": tool_messages}
    
    def _format_tool_response(self, response: dict) -> str:
        """Format Tavily search response as a readable string."""
        results = response.get("results", [])
        logger.debug(f"_format_tool_response: Got {len(results)} results from search")
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "")
            formatted.append(f"{i}. {title}\n   URL: {url}\n   Content: {content}")
        
        return "\n\n".join(formatted)
    
    def _format_math_response(self, result: float) -> str:
        """Format math tool response as a readable string."""
        logger.debug(f"_format_math_response: Formatting result {result}")
        return f"The result is: {result}"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        logger.debug("_build_graph: Creating StateGraph")
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        logger.debug("_build_graph: Adding 'agent' node")
        workflow.add_node("agent", self._call_model)
        logger.debug("_build_graph: Adding 'tools' node")
        workflow.add_node("tools", self._run_tools)
        
        # Set entry point
        logger.debug("_build_graph: Setting entry point to 'agent'")
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        logger.debug("_build_graph: Adding conditional edges from 'agent'")
        workflow.add_conditional_edges(
            "agent",
            self._route_message,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        logger.debug("_build_graph: Adding edge from 'tools' to 'agent'")
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        logger.debug("_build_graph: Compiling graph")
        return workflow.compile()
    
    def _trim_memory(self, messages: Sequence[BaseMessage]) -> list:
        """
        Trim messages to respect memory limit.
        Keeps system message and recent messages up to memory_limit.
        
        Args:
            messages: List of messages to trim
            
        Returns:
            Trimmed list of messages
        """
        logger.debug(f"_trim_memory: Starting with {len(messages)} messages, limit: {self.memory_limit}")
        
        if len(messages) <= self.memory_limit:
            logger.debug("_trim_memory: No trimming needed")
            return list(messages)
        
        # Keep system message at the start
        system_message = None
        remaining_messages = list(messages)
        
        if messages and isinstance(messages[0], SystemMessage):
            system_message = messages[0]
            remaining_messages = messages[1:]
        
        # Calculate how many messages we can keep (excluding system message)
        if system_message:
            max_user_ai_messages = self.memory_limit - 1
        else:
            max_user_ai_messages = self.memory_limit
        
        # Keep only the most recent messages up to the limit
        if len(remaining_messages) > max_user_ai_messages:
            logger.debug(f"_trim_memory: Trimming {len(remaining_messages)} to {max_user_ai_messages} messages")
            remaining_messages = remaining_messages[-max_user_ai_messages:]
        
        # Reconstruct the message list
        if system_message:
            result = [system_message] + remaining_messages
            logger.debug(f"_trim_memory: Result has {len(result)} messages (1 system + {len(remaining_messages)} user/ai)")
            return result
        logger.debug(f"_trim_memory: Result has {len(remaining_messages)} messages")
        return remaining_messages
    
    def _encode_file_to_base64(self, file_path: str) -> tuple:
        """
        Encode a file to base64 with MIME type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (base64_string, mime_type)
        """
        import mimetypes
        
        logger.debug(f"_encode_file_to_base64: Processing file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"_encode_file_to_base64: File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Guess MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # Default to binary for unknown types
            mime_type = "application/octet-stream"
        
        # Read and encode file
        with open(file_path, "rb") as f:
            file_data = f.read()
            base64_data = base64.b64encode(file_data).decode("utf-8")
        
        logger.debug(f"_encode_file_to_base64: File encoded, MIME type: {mime_type}, size: {len(file_data)} bytes")
        return base64_data, mime_type
    
    def _create_message_with_files(self, text: str, file_paths: Optional[List[str]] = None) -> HumanMessage:
        """
        Create a HumanMessage with text and optional file attachments.
        
        Args:
            text: The text content
            file_paths: Optional list of file paths to attach
            
        Returns:
            HumanMessage with text and images/files
        """
        if not file_paths:
            return HumanMessage(content=text)
        
        # Build content parts for multimodal message
        content_parts = [{"type": "text", "text": text}]
        
        for file_path in file_paths:
            try:
                base64_data, mime_type = self._encode_file_to_base64(file_path)
                
                if mime_type.startswith("image/"):
                    # Add image as base64
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_data}"
                        }
                    })
                    logger.debug(f"_create_message_with_files: Added image: {file_path}")
                else:
                    # For non-image files, add as text with file info
                    file_name = os.path.basename(file_path)
                    content_parts.append({
                        "type": "text",
                        "text": f"\n\n[File: {file_name}]\n(Base64 encoded file attached - MIME type: {mime_type})"
                    })
                    logger.debug(f"_create_message_with_files: Added file reference: {file_path}")
            except Exception as e:
                logger.error(f"_create_message_with_files: Error processing file {file_path}: {e}")
                # Add error message for this file
                content_parts.append({
                    "type": "text",
                    "text": f"\n\n[Error: Could not process file '{file_path}': {str(e)}]"
                })
        
        return HumanMessage(content=content_parts)
    
    def run(self, query: str, file_paths: Optional[List[str]] = None) -> str:
        """
        Run the agent with a query and optional file attachments.
        
        Args:
            query: The user's query
            file_paths: Optional list of file paths to attach (images, documents, etc.)
            
        Returns:
            The agent's response
        """
        import time
        
        logger.debug(f"run: Starting with query: '{query[:50]}...' (length: {len(query)})")
        if file_paths:
            logger.debug(f"run: Attached files: {file_paths}")
        
        # Initialize conversation history if not exists
        if not hasattr(self, '_conversation_history'):
            self._conversation_history = []
        
        # Create initial state with conversation history
        initial_messages = self._conversation_history.copy() if self._conversation_history else []
        
        # Add system message if history is empty
        if not initial_messages:
            initial_messages.append(
                SystemMessage(content="You are a helpful AI assistant. You can use search to find up-to-date information.")
            )
        
        # Add user query with optional files
        initial_messages.append(self._create_message_with_files(query, file_paths))
        
        initial_state = {
            "messages": initial_messages,
            "is_final": False,
            "memory_limit": self.memory_limit
        }
        
        logger.debug(f"run: Initial state has {len(initial_messages)} messages")
        
        # Run the graph with retry logic for timeout errors
        max_retries = 2
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"run: Attempt {attempt + 1}/{max_retries} - invoking graph")
                result = self.graph.invoke(initial_state)
                logger.debug(f"run: Graph invocation completed successfully")
                break
            except Exception as e:
                logger.error(f"run: Request failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise e
        
        # Get the final response
        final_message = result["messages"][-1]
        logger.debug(f"run: Final message type: {type(final_message).__name__}, content length: {len(final_message.content) if final_message.content else 0}")
        
        # Update conversation history with all messages from the result
        self._update_conversation_history(result["messages"])
        
        logger.debug("run: Returning final response")
        return final_message.content
    
    def run_stream(self, query: str, file_paths: Optional[List[str]] = None):
        """
        Run the agent with a query and stream the results.
        
        Args:
            query: The user's query
            file_paths: Optional list of file paths to attach (images, documents, etc.)
            
        Yields:
            The agent's responses as they come in
        """
        logger.debug(f"run_stream: Starting with query: '{query[:50]}...' (length: {len(query)})")
        if file_paths:
            logger.debug(f"run_stream: Attached files: {file_paths}")
        
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful AI assistant. You can use search to find up-to-date information."),
                self._create_message_with_files(query, file_paths)
            ],
            "is_final": False
        }
        
        logger.debug("run_stream: Invoking graph.stream()")
        
        # Stream the graph
        for output in self.graph.stream(initial_state):
            logger.debug(f"run_stream: Received output: {type(output).__name__}")
            yield output
    
    def get_memory(self) -> list:
        """
        Get current conversation history from memory.
        
        Returns:
            List of messages in memory
        """
        logger.debug(f"get_memory: Returning {len(self._conversation_history) if hasattr(self, '_conversation_history') else 0} messages from memory")
        if not hasattr(self, '_conversation_history'):
            return []
        return self._conversation_history.copy()
    
    def set_memory_limit(self, limit: int) -> None:
        """
        Set the memory limit for short memory.
        
        Args:
            limit: Maximum number of messages to keep in memory
        """
        logger.debug(f"set_memory_limit: Changing from {self.memory_limit} to {limit}")
        self.memory_limit = limit
        logger.info(f"Memory limit set to {limit} messages")
        
        # Trim existing history if needed
        if hasattr(self, '_conversation_history'):
            self._conversation_history = self._trim_memory(self._conversation_history)
    
    def clear_memory(self) -> None:
        """Clear all conversation history from memory."""
        logger.debug("clear_memory: Clearing conversation history")
        if hasattr(self, '_conversation_history'):
            self._conversation_history = []
            logger.info("Conversation history cleared")
    
    def _update_conversation_history(self, messages: Sequence[BaseMessage]) -> None:
        """
        Update conversation history with new messages.
        
        Args:
            messages: New messages to add to history
        """
        logger.debug(f"_update_conversation_history: Adding {len(messages)} messages to history")
        
        if not hasattr(self, '_conversation_history'):
            self._conversation_history = []
        
        # Add new messages to history
        for msg in messages:
            self._conversation_history.append(msg)
        
        # Trim history to respect memory limit
        self._conversation_history = self._trim_memory(self._conversation_history)
        
        logger.info(f"Conversation history updated. Current size: {len(self._conversation_history)}/{self.memory_limit}")


def main():
    """Main function to run the agent."""
    logger.debug("main: Starting agent")
    # Initialize the agent
    agent = AIAgent()
    logger.debug("main: Agent initialized successfully")
    
    print("AI Agent initialized successfully!")
    print("Type 'quit' or 'exit' to stop.")
    print("To attach a file, type 'file:' followed by the file path (e.g., 'file: ./image.png')")
    print("For multiple files, separate paths with commas (e.g., 'file: ./img1.png, ./img2.jpg')")
    print()
    
    # Chat loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            logger.info("main: User exited the chat")
            break
        
        if not user_input:
            continue
        
        # Parse file attachments
        file_paths = None
        query = user_input
        
        if user_input.lower().startswith("file:"):
            # Extract file paths and query
            rest = user_input[5:].strip()
            if "," in rest:
                # First part might be files, rest is query
                parts = rest.split(",", 1)
                if len(parts) == 2:
                    file_paths = [f.strip() for f in parts[0].split(",")]
                    query = parts[1].strip()
                else:
                    # All are files, no query
                    file_paths = [f.strip() for f in rest.split(",")]
                    query = ""
            else:
                # Single file or just query after "file:"
                if os.path.exists(rest):
                    file_paths = [rest]
                    query = ""
                else:
                    query = rest
        
        if not query and not file_paths:
            print("Please provide a query or file path.")
            continue
        
        print("\nAgent: ", end="", flush=True)
        
        # Get response
        if file_paths:
            logger.info(f"main: Processing request with files: {file_paths}")
            response = agent.run(query, file_paths=file_paths)
        else:
            response = agent.run(query)
        print(response)
        print()  # Empty line for readability


if __name__ == "__main__":
    main()
