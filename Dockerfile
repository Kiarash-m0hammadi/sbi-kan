# SBI-KAN Docker image
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

# Set working directory
WORKDIR /workspace

# Avoid interactive prompts during apt-get / pip
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install git (required to pip install efficient-kan directly from GitHub)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Remove PEP 668 restriction (this is a Docker container, not a host OS)
RUN rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

# Ensure Python always finds your modules when using docker-compose volume mounts
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"

# Copy pyproject.toml AND the src directory so setuptools can find it
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies in editable mode
RUN pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the project (README, notebooks, etc.)
COPY . .

# Default command: interactive bash
CMD ["bash"]