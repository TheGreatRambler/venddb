#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import timm
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "labels_json",
        type=Path,
        help="Path to labels .json file (model .pt must be alongside it)",
    )
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument(
        "--model-id",
        type=str,
        default="convnextv2_large",
        help="timm model ID used during training",
    )
    args = parser.parse_args()

    with open(args.labels_json) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    model = timm.create_model(
        args.model_id, num_classes=len(label_map), pretrained=False
    )
    model.load_state_dict(
        torch.load(
            args.labels_json.with_suffix(".pt"), map_location="cpu", weights_only=True
        )
    )
    model.eval()

    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )

    with torch.no_grad():
        logits = model(transform(Image.open(args.image).convert("RGB")).unsqueeze(0))

    print(idx_to_label[logits.argmax(dim=-1).item()])


if __name__ == "__main__":
    main()
