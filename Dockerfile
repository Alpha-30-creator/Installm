# InstaLLM - OpenAI-compatible API gateway for open-source LLMs
#
# Build (CPU):
#   docker build -t installm .
#
# Build (GPU - requires NVIDIA Container Toolkit):
#   docker build --build-arg BASE=nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 -t installm-gpu .
#
# Run:
#   docker run -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     installm up --model <model-id> --backend transformers

ARG BASE=python:3.11-slim
FROM ${BASE}

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package with transformers extras
RUN pip install --no-cache-dir -e ".[transformers]"

# Expose the default gateway port
EXPOSE 8000

# Default command - can be overridden at runtime
ENTRYPOINT ["installm"]
CMD ["--help"]
