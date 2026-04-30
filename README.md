# AI Agent with LangGraph, RouterAPI, and Tavily

This project implements an AI agent based on LangGraph, using RouterAPI (OpenAI Compatible) as the LLM provider and Tavily for web search.

## Features

- **LangGraph**: Builds a stateful agent with tool-calling capabilities
- **RouterAPI**: Uses RouterAPI's OpenAI-compatible API for LLM inference
- **Tavily**: Integrates Tavily search for up-to-date information retrieval

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys:
     - `ROUTERAPI_BASE_URL`: Your RouterAPI base URL
     - `ROUTERAPI_API_KEY`: Your RouterAPI API key
     - `TAVILY_API_KEY`: Your Tavily API key

## Usage

### Running the Chat Agent

```bash
python agent.py
```

### Using as a Module

```python
from agent import AIAgent

# Initialize the agent
agent = AIAgent()

# Run a query
response = agent.run("What is the current weather in Moscow?")
print(response)
```

### Streaming Responses

```python
for output in agent.run_stream("Tell me about LangGraph"):
    print(output)
```

## Project Structure

```
AiAgent/
├── agent.py          # Main agent implementation
├── requirements.txt  # Python dependencies
├── .env.example      # Example environment variables
├── .env              # Your environment variables (not tracked)
└── README.md         # This file
```

## License

MIT
