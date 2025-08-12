# Start from the base image you were using
FROM runpod/worker-infinity-embedding:1.1.4

# Set the working directory
WORKDIR /

# Update the necessary libraries to their latest versions
# This ensures support for newer models like Qwen3
RUN python3 -m pip install --upgrade pip && \
    pip install --upgrade transformers sentence-transformers torch
