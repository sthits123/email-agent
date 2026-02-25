# 📧 AI Email Assistant

An autonomous, agentic email management system built with **LangGraph** and **GPT-4**. This assistant intelligently triages incoming emails, manages calendar schedules, and drafts personalized responses while maintaining long-term memory of user preferences.

## 🚀 Key Features

- **Intelligent Triage**: Automatically classifies emails into `respond`, `notify`, or `ignore` categories using sophisticated LLM reasoning.
- **Human-in-the-Loop (HITL)**: Native support for manual review of high-stakes actions (sending emails, scheduling meetings) via LangGraph's `interrupt` and `Command` API.
- **Context-Aware Memory**: Persists user preferences and interaction history across threads using LangGraph's state management and custom memory profiles.
- **Seamless Integrations**: Full integration with **Gmail API** for message management and **Google Calendar API** for automated scheduling.
- **Production-Ready**: Optimized for local development with **LangGraph Studio** and containerized for deployment via **Docker** and **LangGraph Cloud**.

## 🛠️ Architecture

The system utilizes a multi-node **StateGraph** architecture:
1. **Triage Router**: Analyzes incoming content against user-defined priorities.
2. **Response Agent**: Orchestrates tool usage and drafts content.
3. **Interrupt Handler**: Manages user input and feedback loops for final approval.

## 📦 Tech Stack

- **Framework**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain)
- **Model**: OpenAI GPT-4
- **Packaging**: [uv](https://github.com/astral-sh/uv)
- **APIs**: Google Workspace (Gmail & Calendar)
- **Infrastructure**: Docker, LangGraph Cloud

## 🚦 Getting Started

### Prerequisites
- Python 3.11+
- `uv` installed
- Google Cloud Project with Gmail/Calendar API enabled

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd email-assistant
   ```
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY
   ```
4. Perform OAuth authentication:
   ```bash
   uv run python3 src/email_assistant/tools/gmail/setup_gmail.py
   ```

### Execution
Run the assistant locally with LangGraph Studio:
```bash
uv run langgraph dev
```

## 📄 License
MIT
