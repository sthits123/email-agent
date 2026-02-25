# Use a multi-stage build or the official uv image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
	build-essential \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# The easy way to get uv in your image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Expose the default LangGraph API port
EXPOSE 8000

# Command to run the agent in dev mode (for now)
CMD ["uv", "run", "langgraph", "dev", "--host", "0.0.0.0", "--port", "8000"]
