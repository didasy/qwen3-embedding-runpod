import os
import time
import runpod
import torch
from sentence_transformers import SentenceTransformer

"""
RunPod Serverless handler for Qwen3-Embedding-8B.

INPUT SCHEMA (examples at bottom):
{
  "input": {
    "texts": ["hello world", "apa kabar?"],
    "as_query": false,            # if true, use prompt_name="query"
    "prompt_name": null,          # optional; overrides as_query if set (e.g., "query")
    "prompt": null,               # optional custom instruction string
    "batch_size": 8,              # 1..64
    "normalize": true,            # cosine-ready
    "truncate_dim": null          # e.g., 1024..4096 (Matryoshka-style)
  }
}

OUTPUT:
{
  "model": "...",
  "device": "cuda"|"cpu",
  "count": <n_texts>,
  "dimension": <embedding_dim>,
  "elapsed_sec": <float>,
  "prompt_name_used": null|"query"|"...",
  "normalized": true|false,
  "embeddings": [[...], ...]
}
"""

# ----------------------------
# Model init (once per pod)
# ----------------------------
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-8B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you need a private HF token, set HF_TOKEN in your RunPod template env.
HF_TOKEN = os.getenv("HF_TOKEN")

# Keep it compatible across GPUs; let Transformers select dtype automatically.
# tokenizer_kwargs lets us ensure left-padding; recommended for many embedding models.
model = SentenceTransformer(
    MODEL_ID,
    device=DEVICE,
    token=HF_TOKEN,
    tokenizer_kwargs={"padding_side": "left"},
    model_kwargs={"trust_remote_code": True},   # harmless if not needed
)

def _encode(
    texts,
    prompt_name=None,
    prompt=None,
    batch_size=8,
    normalize=True,
    truncate_dim=None
):
    # sentence-transformers >= 2.7.0 supports these args.
    return model.encode(
        texts,
        batch_size=max(1, min(int(batch_size), 64)),
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
        prompt_name=prompt_name,
        prompt=prompt,
        truncate_dim=truncate_dim
    )

# ----------------------------
# RunPod handler
# ----------------------------
def handler(job):
    inp = job.get("input", {}) or {}

    texts = inp.get("texts") or inp.get("inputs")
    if not isinstance(texts, (list, tuple)) or len(texts) == 0:
        return {"error": "Missing 'texts' (list of strings) in input."}

    # Options
    as_query = bool(inp.get("as_query", False))
    prompt_name = inp.get("prompt_name", ("query" if as_query else None))
    prompt = inp.get("prompt")
    batch_size = int(inp.get("batch_size", 8))
    normalize = bool(inp.get("normalize", True))

    truncate_dim = inp.get("truncate_dim")
    if truncate_dim is not None:
        try:
            truncate_dim = int(truncate_dim)
        except Exception:
            return {"error": "truncate_dim must be an integer (e.g., 1024..4096)."}

    t0 = time.time()
    vecs = _encode(
        texts=texts,
        prompt_name=prompt_name,
        prompt=prompt,
        batch_size=batch_size,
        normalize=normalize,
        truncate_dim=truncate_dim
    )
    elapsed = round(time.time() - t0, 3)

    # Convert to plain lists for JSON
    vecs_list = [row.tolist() for row in vecs]
    dim = len(vecs_list[0]) if vecs_list else (truncate_dim or 0)

    return {
        "model": MODEL_ID,
        "device": DEVICE,
        "count": len(vecs_list),
        "dimension": dim,
        "elapsed_sec": elapsed,
        "prompt_name_used": prompt_name,
        "normalized": normalize,
        "embeddings": vecs_list
    }

# Start the RunPod worker
runpod.serverless.start({"handler": handler})

# Optional: local test
if __name__ == "__main__":
    ex = {
        "input": {
            "texts": ["hello world", "apa kabar?"],
            "as_query": False,
            "truncate_dim": 4096,
            "normalize": True
        }
    }
    print(handler(ex))
