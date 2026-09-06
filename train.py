"""
train.py

Training loop for the multitask hand gesture model.

Trains on the RealSense data only. The coursework forbids smartphone
images in training for every experiment, so this script never touches
the smartphone set - that is evaluation only.

USAGE
-----
    # Experiment 1 baseline
    python src/train.py --name baseline --epochs 40

    # Experiment 3: stronger photometric augmentation, aimed at
    # closing the RealSense-to-phone domain gap
    python src/train.py --name strong_aug --epochs 40 --augment-strength 1.8

Checkpoints go to weights/<name>_best.pt and weights/<name>_last.pt.
The per-epoch history goes to results/<name>_history.json.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import make_dataloaders, GESTURES, CLASS_NAME
from model import HandGestureNet, MultitaskLoss, box_iou, cxcywh_to_xyxy


# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the multitask gesture model")
    p.add_argument("--data-root", default="rgb_only",
                   help="folder containing the student folders")
    p.add_argument("--name", default="baseline",
                   help="run name, used for checkpoint and log filenames")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32,
                   help="model channel width; lower if you run out of VRAM")
    p.add_argument("--size", type=int, nargs=2, default=[320, 240],
                   metavar=("W", "H"))
    p.add_argument("--augment-strength", type=float, default=1.0,
                   help="scales augmentation probability and magnitude")
    p.add_argument("--domain-aug", action="store_true",
                   help="use the domain-targeted augmentation for Experiment 3: "
                        "scale and colour ranges chosen to span the measured "
                        "RealSense-to-smartphone gap")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device, size) -> Dict[str, float]:
    """
    One pass over the validation set.

    Returns the headline metrics for each task. The full metric suite
    (Dice, macro-F1, confusion matrix) lives in evaluate.py; this is the
    subset worth watching every epoch.
    """
    model.eval()
    w, h = size

    n = 0
    loss_sum = 0.0
    box_iou_sum = 0.0
    box_hits = 0
    mask_inter = 0.0
    mask_union = 0.0
    correct = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = {
            "mask": batch["mask"].to(device, non_blocking=True),
            "box": batch["box"].to(device, non_blocking=True),
            "label": batch["label"].to(device, non_blocking=True),
        }

        out = model(images)
        loss, _ = criterion(out, targets, (w, h))

        b = images.shape[0]
        n += b
        loss_sum += float(loss) * b

        # detection
        pred_box = cxcywh_to_xyxy(out["box_cxcywh"], w, h)
        ious = box_iou(pred_box, targets["box"])
        box_iou_sum += float(ious.sum())
        box_hits += int((ious >= 0.5).sum())

        # segmentation, accumulated over the whole set rather than
        # averaged per image, so large and small hands weigh correctly
        pred_mask = (torch.sigmoid(out["mask_logits"]) > 0.5).float()
        inter = (pred_mask * targets["mask"]).sum()
        union = ((pred_mask + targets["mask"]) > 0).float().sum()
        mask_inter += float(inter)
        mask_union += float(union)

        # classification
        correct += int((out["class_logits"].argmax(1) == targets["label"]).sum())

    return {
        "loss": loss_sum / max(n, 1),
        "box_iou": box_iou_sum / max(n, 1),
        "det_acc@0.5": box_hits / max(n, 1),
        "mask_iou": mask_inter / max(mask_union, 1e-6),
        "cls_acc": correct / max(n, 1),
    }


def score(metrics: Dict[str, float]) -> float:
    """
    Single number for model selection: the mean of the three task
    metrics. Using validation loss instead would let one task's scale
    dominate checkpoint choice.
    """
    return (metrics["det_acc@0.5"] + metrics["mask_iou"] + metrics["cls_acc"]) / 3


# ----------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"        {torch.cuda.get_device_name(0)}")

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    size = (args.size[0], args.size[1])

    train_dl, val_dl = make_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        size=size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        augment_strength=args.augment_strength,
        domain_aug=args.domain_aug,
    )
    print(f"train: {len(train_dl.dataset)} frames, "
          f"{len(train_dl)} batches/epoch")
    print(f"val:   {len(val_dl.dataset)} frames")

    model = HandGestureNet(num_classes=len(GESTURES), width=args.width).to(device)
    criterion = MultitaskLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    start_epoch = 0
    best = -1.0
    history = []

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimiser"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", -1.0)
        history = ck.get("history", [])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {params:,} parameters, width {args.width}")
    print(f"augment strength: {args.augment_strength}")
    print(f"augmentation:     {'domain-targeted' if args.domain_aug else 'generic'}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = {}
        seen = 0

        for i, batch in enumerate(train_dl):
            images = batch["image"].to(device, non_blocking=True)
            targets = {
                "mask": batch["mask"].to(device, non_blocking=True),
                "box": batch["box"].to(device, non_blocking=True),
                "label": batch["label"].to(device, non_blocking=True),
            }

            optimiser.zero_grad(set_to_none=True)
            out = model(images)
            loss, parts = criterion(out, targets, size)
            loss.backward()

            # gradient clipping: the GIoU term can spike early on, when
            # predicted and target boxes do not overlap
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()

            b = images.shape[0]
            seen += b
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v * b

            if i % 20 == 0:
                print(f"\r  epoch {epoch+1}/{args.epochs}  "
                      f"batch {i+1}/{len(train_dl)}  "
                      f"loss {parts['total']:.3f}", end="", flush=True)

        train_parts = {k: v / max(seen, 1) for k, v in running.items()}
        val_metrics = validate(model, val_dl, criterion, device, size)
        scheduler.step()
        elapsed = time.time() - t0

        s = score(val_metrics)
        flag = ""
        if s > best:
            best = s
            flag = "  <- best"
            torch.save({
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best": best,
                "args": vars(args),
                "val_metrics": val_metrics,
                "history": history,
            }, os.path.join(args.weights_dir, f"{args.name}_best.pt"))

        print(f"\r  epoch {epoch+1}/{args.epochs}  "
              f"train {train_parts['total']:.3f}  "
              f"val {val_metrics['loss']:.3f}  "
              f"det@.5 {val_metrics['det_acc@0.5']:.3f}  "
              f"bIoU {val_metrics['box_iou']:.3f}  "
              f"mIoU {val_metrics['mask_iou']:.3f}  "
              f"acc {val_metrics['cls_acc']:.3f}  "
              f"({elapsed:.0f}s){flag}")

        history.append({
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
            "train": train_parts,
            "val": val_metrics,
            "score": s,
        })

        torch.save({
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "args": vars(args),
            "val_metrics": val_metrics,
            "history": history,
        }, os.path.join(args.weights_dir, f"{args.name}_last.pt"))

        with open(os.path.join(args.results_dir,
                               f"{args.name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nbest validation score: {best:.4f}")
    print(f"checkpoint: {args.weights_dir}/{args.name}_best.pt")
    print(f"history:    {args.results_dir}/{args.name}_history.json")


if __name__ == "__main__":
    main()
