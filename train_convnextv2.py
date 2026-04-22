#!/usr/bin/env python3
"""Train ConvNeXt V2 Huge from scratch using image filenames as class labels."""

import argparse
import hashlib
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

import commentjson
import timm
import torch
from packaging.version import Version

_MIN_TIMM = "0.9.0"
if Version(timm.__version__) < Version(_MIN_TIMM):
    raise RuntimeError(
        f"timm {timm.__version__} is too old; convnextv2 requires >= {_MIN_TIMM}. "
        f"Run: pip install --upgrade timm"
    )
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DEFAULT_MODEL_ID = "convnextv2_large"


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

    pip_install = f" && pip install -q {' '.join(pip_packages)}" if pip_packages else ""
    setup = f"apt-get update -qq && apt-get install -y rsync{pip_install}"
    log.info("[remote] Installing dependencies ...")
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
    log.info(f"[remote] {cmd}")
    subprocess.run(ssh_cmd + ["-t", host, cmd], check=True)

    log.info("[remote] Syncing results back ...")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync"]
        + rsync_opts
        + [f"{host}:{rdir}/models_workdir/", f"{local_output_dir}/"],
        check=True,
    )

    sync_dirs: set[Path] = set()
    for flag in ("--output", "--onnx"):
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


def load_config(path: Path) -> dict:
    with open(path) as f:
        return commentjson.load(f)


def build_label_map(data_dir: Path) -> dict[str, int]:
    labels = sorted(
        {
            p.stem
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
    )
    return {label: idx for idx, label in enumerate(labels)}


class ImageFilenameDataset(Dataset):
    def __init__(self, data_dir: Path, label_map: dict[str, int], transform):
        self.label_map = label_map
        self.transform = transform
        self.samples = [
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not self.samples:
            raise ValueError(f"No images found in {data_dir}")
        log.info(
            f"Found {len(self.samples)} images across {len(label_map)} classes in {data_dir}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), self.label_map[path.stem]


def get_core(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt V2 Huge from scratch")
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing training images"
    )
    parser.add_argument("--config", type=Path, default=Path("config.jsonc"))
    parser.add_argument(
        "--model-id", type=str, default=DEFAULT_MODEL_ID, help="timm model ID"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1, metavar="N",
                        help="Accumulate gradients over N micro-batches before each optimizer step")
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Steps between periodic checkpoints",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH/TO/LABELS.json",
        help="Path for the output labels JSON; the .pt (or .onnx) is saved alongside it",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        metavar="PATH/TO/MODEL.onnx",
        help="Export ONNX to this path (+ .yaml alongside); .pt still written to workdir",
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

    cfg = load_config(args.config)
    DATA_DIR = args.data_dir.resolve()
    CONVNEXTV2_MODELS_WORKDIR = Path(cfg["convnextv2_models_dir"]).resolve()

    if args.ssh:
        _run_remote(
            args.ssh,
            Path(__file__).resolve(),
            DATA_DIR,
            args.config.resolve(),
            CONVNEXTV2_MODELS_WORKDIR,
            port=args.ssh_port,
            pip_packages=["commentjson", "timm", "packaging", "transformers", "onnx"],
        )
    dir_hash = hashlib.md5(str(DATA_DIR).encode()).hexdigest()[:12]
    work_dir = CONVNEXTV2_MODELS_WORKDIR / f"convnextv2_workdir_{dir_hash}"

    CONVNEXTV2_MODELS_WORKDIR.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Models dir : {CONVNEXTV2_MODELS_WORKDIR}")
    log.info(f"Work dir   : {work_dir}")
    log.info(f"Data dir   : {DATA_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log.info(f"Device: {device}  AMP: {use_amp}")

    labels_path = work_dir / "labels.json"
    if args.resume and labels_path.exists():
        with open(labels_path) as f:
            label_map = json.load(f)
        log.info(f"Loaded {len(label_map)} labels from {labels_path}")
    else:
        label_map = build_label_map(DATA_DIR)
        with open(labels_path, "w") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        log.info(f"Built and saved {len(label_map)} labels -> {labels_path}")

    num_classes = len(label_map)

    log.info(
        f"Initializing {args.model_id} with {num_classes} classes (random weights)"
    )
    model = timm.create_model(args.model_id, num_classes=num_classes, pretrained=False)
    model.to(device)

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        log.info(f"Using {n_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"Parameters: {total_params:.2f}B")

    data_config = timm.data.resolve_model_data_config(get_core(model))
    train_transform = timm.data.create_transform(**data_config, is_training=True)
    log.info(f"Input size: {data_config['input_size']}")

    dataset = ImageFilenameDataset(DATA_DIR, label_map, train_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_amp,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    global_step = 0
    latest_ckpt = work_dir / "latest_checkpoint.pt"

    if args.resume and latest_ckpt.exists():
        log.info(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        get_core(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        log.info(f"Resumed at epoch {start_epoch}, step {global_step}")

    def save_checkpoint(epoch, step, named=True):
        core = get_core(model)
        state = {
            "model": core.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": step,
        }
        tmp = latest_ckpt.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.rename(latest_ckpt)
        if named:
            named_path = work_dir / f"checkpoint_step{step:08d}.pt"
            torch.save(
                {"model": core.state_dict(), "epoch": epoch, "global_step": step},
                named_path,
            )
            log.info(f"Checkpoint saved -> {named_path}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad()

        for micro_step, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = F.cross_entropy(logits, targets) / args.grad_accum

            scaler.scale(loss).backward()

            is_update_step = (micro_step + 1) % args.grad_accum == 0 or (micro_step + 1) == len(loader)
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.grad_accum
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)

            if is_update_step and global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                acc = correct / total * 100
                log.info(
                    f"epoch {epoch+1}/{args.epochs}  step {global_step}  "
                    f"loss={loss.item() * args.grad_accum:.4f}  acc={acc:.1f}%  lr={lr:.2e}"
                )

            if is_update_step and global_step % args.save_every == 0:
                save_checkpoint(epoch, global_step)

        avg_loss = epoch_loss / len(loader)
        avg_acc = correct / total * 100
        log.info(
            f"Epoch {epoch+1} done — avg loss: {avg_loss:.4f}  acc: {avg_acc:.1f}%"
        )
        save_checkpoint(epoch + 1, global_step, named=False)

    log.info("Training complete.")
    labels_path = (
        args.output.resolve() if args.output else work_dir / "final_model.json"
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    with open(labels_path, "w") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    log.info(f"Labels saved -> {labels_path}")

    core = get_core(model)
    if args.onnx:
        data_config = timm.data.resolve_model_data_config(core)
        input_size = data_config["input_size"][1]
        core.eval().cpu()
        dummy = torch.zeros(1, 3, input_size, input_size)
        onnx_path = args.onnx.resolve()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            core,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
        )
        log.info(f"ONNX model saved -> {onnx_path}")
        yaml_path = onnx_path.with_suffix(".yaml")
        try:
            labels_rel = labels_path.relative_to(yaml_path.parent)
        except ValueError:
            labels_rel = labels_path
        yaml_path.write_text(
            f"model_path: {onnx_path.name}\n"
            f"labels_path: {labels_rel}\n"
            f"input_size: {input_size}\n"
        )
        log.info(f"Inference config saved -> {yaml_path}")

        # Also write .pt + .json to workdir so infer_convnextv2.py works without ONNX runtime
        wd_json = work_dir / "final_model.json"
        wd_pt = work_dir / "final_model.pt"
        if wd_json.resolve() != labels_path:
            with open(wd_json, "w") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
        torch.save(core.state_dict(), wd_pt)
        log.info(f"PT model (Python inference) -> {wd_pt}")
    else:
        pt_path = labels_path.with_suffix(".pt")
        torch.save(core.state_dict(), pt_path)
        log.info(f"Model saved -> {pt_path}")


if __name__ == "__main__":
    main()
