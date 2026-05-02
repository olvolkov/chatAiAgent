# AI Agent with LangGraph, RouterAPI, and Tavily

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://python.langchain.com/)

A stateful AI agent built with LangGraph that can perform web searches, calculate math operations, and provide intelligent responses using RouterAPI (OpenAI Compatible) and Tavily search.

## Features

- **Stateful Agent Architecture**: Built with LangGraph for managing conversation state and memory
- **OpenAI Compatible API**: Uses RouterAPI for LLM inference with support for multiple models
- **Web Search**: Tavily integration for real-time information retrieval
- **Math Tool**: Built-in calculator for arithmetic operations
- **GUI Interface**: Tkinter-based graphical user interface with markdown rendering
- **Console Interface**: CLI-based chat interface for terminal users
- **Config Management**: JSON-based configuration file for easy setup

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Agent                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   LangGraph  │  │  RouterAPI   │  │   Tavily Search  │   │
│  │   State      │  │  (LLM)       │  │   (Web Search)   │   │
│  │   Management │  │              │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                        │                                    │
│                        ▼                                    │
│              ┌──────────────────┐                           │
│              │   Math Tool      │                           │
│              │   (Calculator)   │                           │
│              └──────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

1. **Clone the repository** (or ensure you're in the project directory)

2. **Install the required dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
   
   - Fill in your API keys in `.env`:
     - `ROUTERAPI_BASE_URL`: Your RouterAPI base URL (default: `https://routerai.ru/api/v1`)
     - `ROUTERAPI_API_KEY`: Your RouterAPI API key
     - `MODEL_NAME`: Model name to use (default: `qwen/qwen3.6-flash`)
     - `TAVILY_API_KEY`: Your Tavily API key
     - `TAVILY_MAX_RESULTS`: Maximum search results (default: `5`)
     - `MEMORY_LIMIT`: Conversation history limit (default: `10`)

4. **Alternative: Use config.json**:
   - Edit `config.json` directly with your API keys and settings

## Usage

### Running the Console Chat Agent

```bash
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe .\agent.py
```

The console agent runs in an interactive chat loop where you can:
- Ask questions naturally
- The agent will automatically use web search when needed
- The agent will use math tools for calculations
- Type `exit` or `quit` to end the session
- Qery with file type like: `file: ./image.png, What's in this image?`  

### Running the GUI Chat Agent

```bash
.\.venv\Scripts\activate
.\.venv\Scripts\python.exe .\gui.py
```

The GUI provides:
- Markdown-rendered responses with formatted text, headers, and links
- Clickable links in responses
- Model selection dropdown
- Configuration management
- Scrollable chat history

### Using as a Module

```python
from agent import AIAgent

# Initialize the agent with custom configuration
agent = AIAgent(
    routerapi_base_url="https://routerai.ru/api/v1",
    routerapi_api_key="your-api-key",
    tavily_api_key="your-tavily-key",
    model_name="qwen/qwen3.6-flash",
    memory_limit=10
)

# Run a query
response = agent.run("What is the current weather in Moscow?")

# Or use the GUI
from gui import GUIApp

app = GUIApp()
app.mainloop()
```

## Tools

### Web Search (Tavily)
The agent automatically uses Tavily search when it needs current information or cannot answer from its training data.

### Math Tool
The `add_numbers` tool performs arithmetic operations programmatically:
```python
# The agent will automatically use this for calculations like:
# "What is 25 + 17?" or "Calculate 123 * 45"
```

## Configuration

### Environment Variables (`.env`)
```env
ROUTERAPI_BASE_URL=https://routerai.ru/api/v1
ROUTERAPI_API_KEY=your_routerapi_api_key_here
MODEL_NAME=qwen/qwen3.6-flash
TAVILY_API_KEY=your_tavily_api_key_here
TAVILY_MAX_RESULTS=5
MEMORY_LIMIT=10
```

### Config File (`config.json`)
```json
{
  "routerapi_base_url": "https://routerai.ru/api/v1",
  "routerapi_api_key": "your_api_key",
  "models": ["qwen/qwen3.6-flash", "x-ai/grok-4.3"],
  "active_model": "qwen/qwen3.6-flash",
  "tavily_api_key": "your_tavily_key",
  "tavily_max_results": 5,
  "memory_limit": 10
}
```

## Project Structure

```
AiAgent/
├── agent.py          # Main agent implementation with LangGraph
├── gui.py            # GUI application with markdown support
├── config.json       # Configuration file
├── .env              # Environment variables (not included)
├── .env.example      # Example environment variables
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── agent.log         # Agent execution logs
├── gui.log           # GUI execution logs
└── tools/
    ├── __init__.py
    └── math_tool.py  # Custom math calculation tool
```

## Requirements

- Python 3.9+
- RouterAPI account and API key
- Tavily account and API key

## License

This project is provided as-is for educational and development purposes.
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
