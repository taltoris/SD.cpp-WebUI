# Use NVIDIA CUDA runtime image with Ubuntu 24.04
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create and activate Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages within the virtual environment
COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Create necessary directories
RUN mkdir -p /app/models

# Set working directory
WORKDIR /app

# Expose ports
EXPOSE 5000 8080

# Run the application using the virtual environment
CMD ["/opt/venv/bin/python", "app.py"]
