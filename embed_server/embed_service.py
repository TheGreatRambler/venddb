#!/usr/bin/env python3
import base64
import io
import commentjson

import numpy as np
import torch
from flask import Flask, Response, request
from PIL import Image
from transformers import AutoModel, AutoProcessor

_CONFIG_PATH = "/app/config.jsonc"


def _load_config():
    try:
        with open(_CONFIG_PATH) as f:
            return commentjson.load(f)
    except Exception:
        print("Could not load config, using defaults.", flush=True)
        return {}


_cfg = _load_config()
# Model is downloaded during Docker build and available at this path.
MODEL_ID = "/app/model"
PORT = int(_cfg.get("embed_service_port", 8000))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

print(f"Loading {MODEL_ID} on {device}...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype).to(device)
model.eval()
print("Model loaded.", flush=True)

app = Flask(__name__)


@app.route("/health")
def health():
    return "ok"


@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True)
    image_bytes = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt", padding="max_length").to(
        device
    )
    with torch.no_grad():
        vec = model.get_image_features(pixel_values=inputs["pixel_values"])

    # For some siglip2 variants get_image_features returns a BaseModelOutputWithPooling
    # rather than a raw tensor; extract the pooled output in that case.
    if not isinstance(vec, torch.Tensor):
        vec = vec.pooler_output

    # Flatten to 1D float32 (len 1536) and return as packed little-endian bytes.
    flat = vec.flatten().cpu().float().numpy().astype(np.float32)
    return Response(flat.tobytes(), mimetype="application/octet-stream")


# Not used by Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
