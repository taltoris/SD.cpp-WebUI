# Vulkan / AMD runtime, Ubuntu 24.04
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python app requirements plus Vulkan loader and Mesa RADV ICD
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    libglib2.0-0 \
    wget \
    ca-certificates \
    libstdc++6 \
    libgomp1 \
    libvulkan1 \
    mesa-vulkan-drivers \
    libdrm2 \
    libdrm-amdgpu1 \
    vulkan-tools \
 && rm -rf /var/lib/apt/lists/*

# Create and activate Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt /tmp/
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Create directories expected by the app/bind mounts
RUN mkdir -p /app/models /app/output

WORKDIR /app

EXPOSE 5000 8080

CMD ["/opt/venv/bin/python", "app.py"]
