# Start from the base image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV HF_HOME=/runpod-volume

# Install Python and other necessary packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    libgl1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Install core packages in separate steps to avoid dependency conflicts
RUN pip install "runpod>=1.7.0, <1.8.0" --no-cache-dir
RUN pip install infinity-emb[all]==0.0.76 --no-cache-dir
RUN pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/test/cu124 --no-cache-dir
RUN pip install transformers>=4.42.0 sentence-transformers einops --no-cache-dir
RUN pip install git+https://github.com/pytorch-labs/float8_experimental.git --no-cache-dir

# Add src files
ADD src .

# Add test input
COPY test_input.json /test_input.json

# start the handler
CMD python -u /handler.py
