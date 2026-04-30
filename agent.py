"""
AI Agent based on LangGraph with RouterAPI (OpenAI Compatible) and Tavily search.
"""

import os
from typing import TypedDict, Annotated, Sequence
from operator import add

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], add]
    is_final: bool


class AIAgent:
    """AI Agent using LangGraph with RouterAPI and Tavily search."""
    
    def __init__(
        self,
        routerapi_base_url: str = None,
        routerapi_api_key: str = None,
        tavily_api_key: str = None,
        model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize the agent.
        
        Args:
            routerapi_base_url: Base URL for RouterAPI (OpenAI Compatible)
            routerapi_api_key: API key for RouterAPI
            tavily_api_key: API key for Tavily search
            model_name: Model name to use
        """
        # Get from environment if not provided
        self.routerapi_base_url = routerapi_base_url or os.getenv("ROUTERAPI_BASE_URL")
        self.routerapi_api_key = routerapi_api_key or os.getenv("ROUTERAPI_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.model_name = model_name
        
        if not all([self.routerapi_base_url, self.routerapi_api_key, self.tavily_api_key]):
            raise ValueError(
                "Missing required environment variables. "
                "Please set ROUTERAPI_BASE_URL, ROUTERAPI_API_KEY, and TAVILY_API_KEY."
            )
        
        # Initialize LLM with RouterAPI (OpenAI Compatible)
        self.llm = ChatOpenAI(
            base_url=self.routerapi_base_url,
            api_key=self.routerapi_api_key,
            model=self.model_name,
            temperature=0.7
        )
        
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(
            api_key=self.tavily_api_key,
            max_results=5
        )
        
        # Define tools
        self.tools = [self.search_tool]
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _add_messages(self, state: AgentState, new_messages: Sequence[BaseMessage]) -> AgentState:
        """Add new messages to the state."""
        return {"messages": new_messages}
    
    def _call_model(self, state: AgentState) -> AgentState:
        """Call the LLM to generate a response."""
        messages = state["messages"]
        
        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def _route_message(self, state: AgentState) -> str:
        """Determine if we need to use a tool or if we're done."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, we need to use tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, we're done
        return "end"
    
    def _run_tools(self, state: AgentState) -> AgentState:
        """Run the tools based on tool calls in the last message."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # Execute tool calls
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"].lower().replace(" ", "_")
            tool_args = tool_call["args"]
            
            # Find and execute the tool
            selected_tool = None
            for tool in self.tools:
                if tool.name.lower() == tool_name:
                    selected_tool = tool
                    break
            
            if selected_tool:
                tool_response = selected_tool.invoke(tool_args)
                tool_messages.append(tool_response)
        
        # Return messages with tool results
        return {"messages": tool_messages}
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._run_tools)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._route_message,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def run(self, query: str) -> str:
        """
        Run the agent with a query.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response
        """
        # Initialize messages with the query
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful AI assistant. You can use search to find up-to-date information."),
                HumanMessage(content=query)
            ],
            "is_final": False
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Get the final response
        final_message = result["messages"][-1]
        
        return final_message.content
    
    def run_stream(self, query: str):
        """
        Run the agent with a query and stream the results.
        
        Args:
            query: The user's query
            
        Yields:
            The agent's responses as they come in
        """
        initial_state = {
            "messages": [
                SystemMessage(content="You are a helpful AI assistant. You can use search to find up-to-date information."),
                HumanMessage(content=query)
            ],
            "is_final": False
        }
        
        # Stream the graph
        for output in self.graph.stream(initial_state):
            yield output


def main():
    """Main function to run the agent."""
    # Initialize the agent
    agent = AIAgent()
    
    print("AI Agent initialized successfully!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Chat loop
    while True:
        user_query = input("You: ").strip()
        
        if user_query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        print("\nAgent: ", end="", flush=True)
        
        # Get response
        response = agent.run(user_query)
        print(response)
        print()  # Empty line for readability


if __name__ == "__main__":
    main()
