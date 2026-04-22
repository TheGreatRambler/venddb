"""
Run inference with a trained YOLO26 model and save a debug image.

Usage:
    python infer.py <image_path> [--weights path/to/best.pt] [--conf 0.25] [--out output.jpg]
"""

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "runs/vending/weights/best.pt"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to input image")
parser.add_argument(
    "--weights", default=str(DEFAULT_WEIGHTS), help="Path to model weights"
)
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
parser.add_argument(
    "--out", default=None, help="Output image path (default: <stem>_debug.jpg)"
)
args = parser.parse_args()

img_path = Path(args.image)
out_path = Path(args.out) if args.out else Path.cwd() / (img_path.stem + "_debug.jpg")

# ---------------------------------------------------------------------------
# Load model + run inference
# ---------------------------------------------------------------------------

model = YOLO(args.weights)
results = model.predict(str(img_path), conf=args.conf, imgsz=640, verbose=False)[0]

# ---------------------------------------------------------------------------
# Draw boxes
# ---------------------------------------------------------------------------

img = cv2.imread(str(img_path))
assert img is not None, f"Could not read image: {img_path}"

class_names = results.names  # dict {id: name}

# Assign a consistent colour per class
rng = random.Random(42)
colours = {cid: tuple(rng.randint(60, 230) for _ in range(3)) for cid in class_names}

overlay = img.copy()

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cid = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{class_names[cid]} {conf:.2f}"
    colour = colours[cid]

    # Translucent fill
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)

cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cid = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{class_names[cid]} {conf:.2f}"
    colour = colours[cid]

    # Solid border
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

    # Diagonal cross
    cv2.line(img, (x1, y1), (x2, y2), colour, 2)
    cv2.line(img, (x2, y1), (x1, y2), colour, 2)

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(img.shape[1], img.shape[0]) / 2000)
    thickness = max(1, int(scale * 2))
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    ty = max(y1 - 4, th + baseline)
    cv2.rectangle(img, (x1, ty - th - baseline), (x1 + tw, ty + baseline), colour, -1)
    cv2.putText(
        img, label, (x1, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

cv2.imwrite(str(out_path), img)
print(f"Saved debug image → {out_path}  ({len(results.boxes)} detections)")
