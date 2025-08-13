# Newer CUDA/PyTorch from the official PyTorch images
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Qwen3-Embedding-8B needs transformers>=4.51 & sentence-transformers>=2.7
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
        "transformers>=4.51.0" \
        "sentence-transformers>=2.7.0" \
        "accelerate>=0.33.0" \
        "runpod>=1.7.0" \
        "numpy"

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
