"""
Train YOLO11 on vending machine data annotated in LabelMe format.

Usage:
    pip install ultralytics pillow
    python train_yolo.py <input_dir> [--onnx <output.onnx>] [--yaml <output.yaml>]
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import commentjson

from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _run_remote(
    ssh: str,
    script: Path,
    data_dir: Path,
    config_path: Path,
    local_output_dir: Path,
    port: int | None = None,
    pip_packages: list[str] | None = None,
) -> None:
    host, _, rdir = ssh.partition(":")
    rdir = rdir or "~/train_remote"

    ssh_cmd = ["ssh"] + (["-p", str(port)] if port else [])
    rsync_opts = ["-avz", "--update"] + (["-e", f"ssh -p {port}"] if port else [])

    pip_install = (
        f" && pip install -q --break-system-packages {' '.join(pip_packages)}"
        if pip_packages
        else ""
    )
    setup = f"apt-get update -qq && apt-get install -y rsync{pip_install}"
    print("[remote] Installing dependencies ...")
    subprocess.run(ssh_cmd + [host, setup], check=True)

    subprocess.run(ssh_cmd + [host, f"mkdir -p {rdir}/data"], check=True)
    subprocess.run(
        ["rsync"] + rsync_opts + [f"{data_dir}/", f"{host}:{rdir}/data/"], check=True
    )
    subprocess.run(
        ["rsync"] + rsync_opts + [str(script), str(config_path), f"{host}:{rdir}/"],
        check=True,
    )

    remote_args: list[str] = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--ssh", "--ssh-port"):
            i += 2
            continue
        try:
            if Path(a).resolve() == data_dir:
                remote_args.append("data")
                i += 1
                continue
        except Exception:
            pass
        if a == "--config" and i + 1 < len(argv):
            remote_args += ["--config", "./config.jsonc"]
            i += 2
            continue
        try:
            if Path(a).resolve() == config_path:
                remote_args.append("./config.jsonc")
                i += 1
                continue
        except Exception:
            pass
        remote_args.append(a)
        i += 1

    cmd = "cd {} && PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 {} {}".format(
        rdir,
        shlex.quote(script.name),
        " ".join(shlex.quote(a) for a in remote_args),
    )
    print(f"[remote] {cmd}")
    subprocess.run(ssh_cmd + ["-t", host, cmd], check=True)

    print("[remote] Syncing results back ...")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync"]
        + rsync_opts
        + [f"{host}:{rdir}/models_workdir/", f"{local_output_dir}/"],
        check=True,
    )

    sync_dirs: set[Path] = set()
    for flag in ("--onnx", "--yaml"):
        try:
            d = Path(remote_args[remote_args.index(flag) + 1]).parent
            if str(d) != ".":
                sync_dirs.add(d)
        except (ValueError, IndexError):
            pass
    for d in sync_dirs:
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync"] + rsync_opts + [f"{host}:{rdir}/{d}/", f"{d}/"], check=True
        )

    sys.exit(0)


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_dir", type=Path, help="Directory containing LabelMe JSON + images"
)
parser.add_argument(
    "--onnx",
    type=Path,
    default=None,
    help="Output ONNX path (default: same location/name as --yaml)",
)
parser.add_argument(
    "--yaml", type=Path, default=Path("model.yaml"), help="Output YAML path"
)
parser.add_argument("--model", type=str, default="yolo26x.pt", help="Model type/size")
parser.add_argument(
    "--config", type=Path, default=Path("config.jsonc"), help="Path to config.jsonc"
)
parser.add_argument(
    "--ssh",
    type=str,
    default=None,
    metavar="[user@]host[:path]",
    help="Run on remote GPU: rsync files and execute via SSH",
)
parser.add_argument(
    "--ssh-port",
    type=int,
    default=None,
    metavar="PORT",
    help="SSH port for --ssh (default: 22)",
)
args = parser.parse_args()

DATA_DIR = args.input_dir.resolve()
YAML_OUT = args.yaml.resolve()
ONNX_OUT = (args.onnx or YAML_OUT.with_suffix(".onnx")).resolve()

with args.config.open() as _f:
    _config = commentjson.load(_f)
WORK_DIR_BASE = Path(_config["yolo_models_dir"]).resolve()

if args.ssh:
    _run_remote(
        args.ssh,
        Path(__file__).resolve(),
        DATA_DIR,
        args.config.resolve(),
        WORK_DIR_BASE,
        port=args.ssh_port,
        pip_packages=["commentjson", "ultralytics"],
    )

# Stable working directory per input, keyed by normalized path hash
dir_hash = hashlib.md5(str(DATA_DIR).encode()).hexdigest()[:12]
WORK_DIR = WORK_DIR_BASE / f"yolo_workdir_{dir_hash}"
DATASET_DIR = WORK_DIR / "dataset"

MODEL_PATH = (
    WORK_DIR / args.model
)  # n / s / m / l / x. Must be one of the valid sizes, as well as a path
EPOCHS = 300
IMG_SIZE = 640
BATCH = 8
VAL_SPLIT = 0.15  # fraction of images held out for validation

# ---------------------------------------------------------------------------
# Collect all unique class labels
# ---------------------------------------------------------------------------

json_files = sorted(DATA_DIR.glob("*.json"))
assert json_files, f"No JSON files found in {DATA_DIR}"

all_labels: set[str] = set()
for jf in json_files:
    data = json.loads(jf.read_text())
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "rectangle":
            all_labels.add(shape["label"])

class_names = sorted(all_labels)
label_to_id = {name: i for i, name in enumerate(class_names)}
print(f"Classes ({len(class_names)}): {class_names}")

# ---------------------------------------------------------------------------
# Build train / val split
# ---------------------------------------------------------------------------

import random

random.shuffle(json_files)
n_val = max(1, round(len(json_files) * VAL_SPLIT))
val_set = set(jf.stem for jf in json_files[:n_val])

for split in ("train", "val"):
    (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Convert LabelMe → YOLO label files and copy images
# ---------------------------------------------------------------------------


def find_image(stem: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".png", ".PNG", ".jpeg", ".JPEG"):
        p = DATA_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def labelme_rect_to_yolo(
    points: list, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert 4-corner LabelMe rectangle to YOLO cx, cy, w, h (normalised)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    return cx, cy, bw, bh


skipped = 0
for jf in json_files:
    data = json.loads(jf.read_text())
    img_path = find_image(jf.stem)
    if img_path is None:
        print(f"  [warn] no image for {jf.name}, skipping")
        skipped += 1
        continue

    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if not (img_w and img_h):
        img_w, img_h = Image.open(img_path).size

    split = "val" if jf.stem in val_set else "train"

    # Copy image
    dst_img = DATASET_DIR / "images" / split / img_path.name
    if not dst_img.exists():
        shutil.copy2(img_path, dst_img)

    # Write label file
    lines = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        class_id = label_to_id[shape["label"]]
        cx, cy, bw, bh = labelme_rect_to_yolo(shape["points"], img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    label_file = DATASET_DIR / "labels" / split / (jf.stem + ".txt")
    label_file.write_text("\n".join(lines))

print(
    f"Prepared dataset  train={len(json_files) - n_val - skipped}  val={n_val}  skipped={skipped}"
)

# ---------------------------------------------------------------------------
# Write dataset YAML (model_path as relative path from YAML location)
# ---------------------------------------------------------------------------

try:
    model_rel = Path(ONNX_OUT).relative_to(YAML_OUT.parent)
except ValueError:
    model_rel = ONNX_OUT  # fall back to absolute if not under the same tree

classes_yaml = "\n".join(f"  - {n}" for n in class_names)
yaml_content = f"""path: {DATASET_DIR.resolve()}
train: images/train
val:   images/val

nc: {len(class_names)}
names: {class_names}

type: yolo26
name: yolo26x
provider: Ultralytics
display_name: YOLO26x
model_path: {model_rel}
iou_threshold: 0.60
conf_threshold: 0.25
agnostic: True
classes:
{classes_yaml}
"""
YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
YAML_OUT.write_text(yaml_content)
print(f"Dataset YAML: {YAML_OUT}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

model = YOLO(MODEL_PATH)
result = model.train(
    data=str(YAML_OUT),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    project=str(WORK_DIR / "runs"),
    name="vending",
)

# ---------------------------------------------------------------------------
# Export best weights to ONNX
# ---------------------------------------------------------------------------

best_pt = Path(result.save_dir) / "weights/best.pt"
trained = YOLO(str(best_pt))
trained.export(format="onnx", imgsz=IMG_SIZE)

onnx_src = best_pt.with_suffix(".onnx")
ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(onnx_src, ONNX_OUT)
print(f"ONNX model → {ONNX_OUT}")
