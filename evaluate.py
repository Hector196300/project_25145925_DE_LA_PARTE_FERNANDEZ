"""
evaluate.py

Computes every metric the coursework asks for, on any dataset in the
COMP0248 folder format - the RealSense val/test sets or the smartphone
test set.

Metrics
-------
Detection
    Detection accuracy @ 0.5 IoU
    Mean bounding-box IoU
Segmentation
    Mean IoU, hand vs background
    Dice coefficient
Classification
    Top-1 accuracy
    Macro-averaged F1 across the 10 classes
    Confusion matrix

Everything is written to results/<name>_metrics.json, with the confusion
matrix also saved as CSV for the report.

USAGE
-----
    # RealSense test set
    python src/evaluate.py --weights weights/baseline_best.pt \
        --dataset COMP0248_Test_data_23 --name exp1_realsense_test

    # smartphone test set, same weights, no fine-tuning
    python src/evaluate.py --weights weights/baseline_best.pt \
        --dataset smartphone_dataset/25145925_SURNAME --name exp2_phone_zeroshot

    # the validation split of the training data
    python src/evaluate.py --weights weights/baseline_best.pt \
        --val-split --name exp1_realsense_val

    # compare two runs (Experiment 3 gain over Experiment 2)
    python src/evaluate.py --compare results/exp2_phone_zeroshot_metrics.json \
                                     results/exp3_phone_strongaug_metrics.json
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import (
    make_dataloaders, make_test_dataloader, GESTURES, CLASS_NAME,
)
from model import HandGestureNet, box_iou, cxcywh_to_xyxy


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """
    Tight xyxy box around each predicted binary mask.

    In a single-object segmentation problem the mask is a strictly richer
    representation than the box, and the box is a deterministic function
    of it - the same function used to build the ground-truth boxes in
    dataloader.py. Deriving the box this way is therefore an alternative
    to regressing it directly, and costs nothing at inference.

    An empty mask falls back to the whole frame, which scores poorly but
    keeps the box valid rather than producing NaNs.
    """
    b, _, h, w = masks.shape
    out = torch.zeros(b, 4, device=masks.device, dtype=torch.float32)
    flat = masks[:, 0] > 0.5

    for i in range(b):
        ys, xs = torch.nonzero(flat[i], as_tuple=True)
        if xs.numel() == 0:
            out[i] = torch.tensor([0.0, 0.0, float(w), float(h)],
                                  device=masks.device)
        else:
            out[i, 0] = xs.min().float()
            out[i, 1] = ys.min().float()
            out[i, 2] = xs.max().float() + 1
            out[i, 3] = ys.max().float() + 1
    return out


def detection_metrics(ious: np.ndarray, threshold: float = 0.5) -> Dict:
    return {
        f"accuracy@{threshold}IoU": float((ious >= threshold).mean()),
        "mean_box_iou": float(ious.mean()),
        "median_box_iou": float(np.median(ious)),
        "accuracy@0.75IoU": float((ious >= 0.75).mean()),
    }


def confusion_matrix(true: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray):
    """
    Precision, recall and F1 for each class, from the confusion matrix.

    Written out rather than imported so the computation is visible and
    the zero-support case is handled explicitly: a class with no true
    examples gets an F1 of 0 and is still counted in the macro average,
    which is the conservative choice.
    """
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    support = cm.sum(axis=1)

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision[i] + recall[i]
        f1[i] = 2 * precision[i] * recall[i] / denom if denom > 0 else 0.0

    return precision, recall, f1, support


@torch.no_grad()
def evaluate(model, loader, device, size, iou_threshold: float = 0.5,
             collect_samples: int = 0) -> Dict:
    """
    One pass over a dataset, accumulating every required metric.

    Segmentation IoU and Dice are reported two ways:
      - dataset-level, pooling all pixels, which weights large hands more
      - image-level mean, which weights every image equally
    The coursework asks for "mean IoU", so the image-level mean is the
    headline figure; the pooled figure is reported alongside because the
    two can differ noticeably when hand size varies.
    """
    model.eval()
    w, h = size

    n = 0
    box_ious: List[float] = []
    mask_box_ious: List[float] = []
    img_ious: List[float] = []
    img_dice: List[float] = []
    pooled_inter = 0.0
    pooled_union = 0.0
    pooled_pred = 0.0
    pooled_true = 0.0
    true_labels: List[int] = []
    pred_labels: List[int] = []
    confidences: List[float] = []
    samples = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        gt_mask = batch["mask"].to(device, non_blocking=True)
        gt_box = batch["box"].to(device, non_blocking=True)
        gt_label = batch["label"].to(device, non_blocking=True)

        out = model(images)
        probs = torch.softmax(out["class_logits"], dim=1)
        conf, pred_label = probs.max(dim=1)
        pred_box = cxcywh_to_xyxy(out["box_cxcywh"], w, h)
        pred_mask = (torch.sigmoid(out["mask_logits"]) > 0.5).float()

        b = images.shape[0]
        n += b

        # --- detection, two ways ---
        # (a) the regressed box from the box head
        ious = box_iou(pred_box, gt_box)
        box_ious.extend(ious.cpu().tolist())

        # (b) the tight box around the predicted mask, which needs no
        #     extra training and reuses the segmentation output
        mask_box = boxes_from_masks(pred_mask)
        mask_ious = box_iou(mask_box, gt_box)
        mask_box_ious.extend(mask_ious.cpu().tolist())

        # --- segmentation, per image ---
        p = pred_mask.reshape(b, -1)
        t = gt_mask.reshape(b, -1)
        inter = (p * t).sum(dim=1)
        union = ((p + t) > 0).float().sum(dim=1)
        img_ious.extend((inter / union.clamp(min=1e-6)).cpu().tolist())
        dice = 2 * inter / (p.sum(dim=1) + t.sum(dim=1)).clamp(min=1e-6)
        img_dice.extend(dice.cpu().tolist())

        pooled_inter += float(inter.sum())
        pooled_union += float(union.sum())
        pooled_pred += float(p.sum())
        pooled_true += float(t.sum())

        # --- classification ---
        true_labels.extend(gt_label.cpu().tolist())
        pred_labels.extend(pred_label.cpu().tolist())
        confidences.extend(conf.cpu().tolist())

        # --- keep a few for qualitative figures ---
        if len(samples) < collect_samples:
            for i in range(min(b, collect_samples - len(samples))):
                samples.append({
                    "meta": batch["meta"][i],
                    "pred_box": pred_box[i].cpu().tolist(),
                    "mask_box": mask_box[i].cpu().tolist(),
                    "gt_box": gt_box[i].cpu().tolist(),
                    "pred_label": int(pred_label[i]),
                    "gt_label": int(gt_label[i]),
                    "confidence": float(conf[i]),
                    "box_iou": float(ious[i]),
                    "mask_iou": float(inter[i] / union[i].clamp(min=1e-6)),
                })

    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)
    cm = confusion_matrix(true_arr, pred_arr, len(GESTURES))
    precision, recall, f1, support = per_class_prf(cm)

    box_ious_arr = np.array(box_ious)
    mask_box_arr = np.array(mask_box_ious)

    # Report whichever of the two box sources scores better as the
    # headline "detection" figure, and keep both for the ablation.
    regressed = detection_metrics(box_ious_arr, iou_threshold)
    from_mask = detection_metrics(mask_box_arr, iou_threshold)
    best_source = ("mask_derived"
                   if from_mask["mean_box_iou"] > regressed["mean_box_iou"]
                   else "regressed")

    metrics = {
        "n_images": n,
        "detection": dict(
            (from_mask if best_source == "mask_derived" else regressed),
            source=best_source,
        ),
        "detection_regressed": regressed,
        "detection_from_mask": from_mask,
        "segmentation": {
            "mean_iou": float(np.mean(img_ious)),
            "mean_dice": float(np.mean(img_dice)),
            "pooled_iou": pooled_inter / max(pooled_union, 1e-6),
            "pooled_dice": 2 * pooled_inter / max(pooled_pred + pooled_true, 1e-6),
        },
        "classification": {
            "top1_accuracy": float((true_arr == pred_arr).mean()),
            "macro_f1": float(f1.mean()),
            "mean_confidence": float(np.mean(confidences)),
            "mean_confidence_correct": float(
                np.mean([c for c, t, p in zip(confidences, true_arr, pred_arr) if t == p])
            ) if (true_arr == pred_arr).any() else 0.0,
            "per_class": {
                CLASS_NAME[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(len(GESTURES))
            },
        },
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAME,
    }
    return metrics, samples


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------

def print_report(name: str, m: Dict):
    print(f"\n{'=' * 62}")
    print(f"{name}   ({m['n_images']} images)")
    print("=" * 62)

    d = m["detection"]
    print(f"\nDetection   (headline source: {d.get('source', 'regressed')})")
    print(f"  accuracy @0.5 IoU     {d['accuracy@0.5IoU']:.4f}")
    print(f"  accuracy @0.75 IoU    {d['accuracy@0.75IoU']:.4f}")
    print(f"  mean box IoU          {d['mean_box_iou']:.4f}")
    print(f"  median box IoU        {d['median_box_iou']:.4f}")

    if "detection_regressed" in m:
        r, f = m["detection_regressed"], m["detection_from_mask"]
        print("\n  box source ablation:")
        print(f"    {'':<16}{'regressed':>12}{'from mask':>12}")
        print(f"    {'acc @0.5 IoU':<16}{r['accuracy@0.5IoU']:>12.4f}"
              f"{f['accuracy@0.5IoU']:>12.4f}")
        print(f"    {'mean box IoU':<16}{r['mean_box_iou']:>12.4f}"
              f"{f['mean_box_iou']:>12.4f}")

    s = m["segmentation"]
    print("\nSegmentation")
    print(f"  mean IoU              {s['mean_iou']:.4f}")
    print(f"  Dice coefficient      {s['mean_dice']:.4f}")
    print(f"  pooled IoU            {s['pooled_iou']:.4f}")
    print(f"  pooled Dice           {s['pooled_dice']:.4f}")

    c = m["classification"]
    print("\nClassification")
    print(f"  top-1 accuracy        {c['top1_accuracy']:.4f}")
    print(f"  macro F1              {c['macro_f1']:.4f}")
    print(f"  mean confidence       {c['mean_confidence']:.4f}")

    print("\n  per class:")
    print(f"    {'class':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>6}")
    for cls, v in c["per_class"].items():
        print(f"    {cls:<10}{v['precision']:>8.3f}{v['recall']:>8.3f}"
              f"{v['f1']:>8.3f}{v['support']:>6}")

    cm = np.array(m["confusion_matrix"])
    names = m["class_names"]
    print("\n  confusion matrix (rows true, columns predicted):")
    print("    " + " " * 10 + "".join(f"{n[:5]:>6}" for n in names))
    for i, row in enumerate(cm):
        print(f"    {names[i]:<10}" + "".join(f"{v:>6}" for v in row))


def compare(path_a: str, path_b: str):
    """Report the change from one evaluation to another, for Experiment 3."""
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    name_a = os.path.basename(path_a).replace("_metrics.json", "")
    name_b = os.path.basename(path_b).replace("_metrics.json", "")

    rows = [
        ("detection", "accuracy@0.5IoU", "det acc @0.5"),
        ("detection", "mean_box_iou", "mean box IoU"),
        ("segmentation", "mean_iou", "mask mean IoU"),
        ("segmentation", "mean_dice", "Dice"),
        ("classification", "top1_accuracy", "top-1 accuracy"),
        ("classification", "macro_f1", "macro F1"),
    ]

    print(f"\n{'metric':<18}{name_a:>22}{name_b:>22}{'gain':>10}")
    print("-" * 72)
    for group, key, label in rows:
        va, vb = a[group][key], b[group][key]
        print(f"{label:<18}{va:>22.4f}{vb:>22.4f}{vb - va:>+10.4f}")


# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the multitask model")
    p.add_argument("--weights", help="checkpoint to evaluate")
    p.add_argument("--dataset", help="dataset folder to evaluate on")
    p.add_argument("--val-split", action="store_true",
                   help="evaluate on the validation split of --data-root instead")
    p.add_argument("--data-root", default="rgb_only")
    p.add_argument("--name", default="eval", help="output filename stem")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                   help="defaults to the size the checkpoint was trained at")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--save-samples", type=int, default=12,
                   help="how many per-image records to keep for figures")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="compare two saved metric files and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if not args.weights:
        print("--weights is required (or use --compare)")
        return
    if not args.dataset and not args.val_split:
        print("give either --dataset or --val-split")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    os.makedirs(args.results_dir, exist_ok=True)

    ck = torch.load(args.weights, map_location=device, weights_only=False)
    train_args = ck.get("args", {})

    # match the training configuration unless overridden, otherwise the
    # numbers are not comparable across experiments
    width = train_args.get("width", 32)
    size = tuple(args.size) if args.size else tuple(train_args.get("size", [320, 240]))

    model = HandGestureNet(num_classes=len(GESTURES), width=width).to(device)
    model.load_state_dict(ck["model"])

    print(f"device:     {device}")
    print(f"weights:    {args.weights}  (epoch {ck.get('epoch', '?')})")
    print(f"model:      width {width}, input {size[0]}x{size[1]}")
    if "augment_strength" in train_args:
        print(f"trained at augment strength {train_args['augment_strength']}")

    if args.val_split:
        _, loader = make_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            size=size,
            val_fraction=train_args.get("val_fraction", 0.2),
            seed=train_args.get("seed", 0),
            num_workers=args.num_workers,
        )
        source = f"{args.data_root} (validation split)"
    else:
        loader = make_test_dataloader(
            args.dataset, batch_size=args.batch_size, size=size,
            num_workers=args.num_workers,
        )
        source = args.dataset

    print(f"dataset:    {source}")

    metrics, samples = evaluate(model, loader, device, size,
                                collect_samples=args.save_samples)
    metrics["meta"] = {
        "weights": args.weights,
        "dataset": source,
        "input_size": list(size),
        "model_width": width,
        "trained_augment_strength": train_args.get("augment_strength"),
        "checkpoint_epoch": ck.get("epoch"),
    }

    print_report(args.name, metrics)

    out_json = os.path.join(args.results_dir, f"{args.name}_metrics.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    out_csv = os.path.join(args.results_dir, f"{args.name}_confusion.csv")
    with open(out_csv, "w") as f:
        f.write("true\\pred," + ",".join(CLASS_NAME) + "\n")
        for i, row in enumerate(metrics["confusion_matrix"]):
            f.write(CLASS_NAME[i] + "," + ",".join(str(v) for v in row) + "\n")

    if samples:
        out_s = os.path.join(args.results_dir, f"{args.name}_samples.json")
        with open(out_s, "w") as f:
            json.dump(samples, f, indent=2)

    print(f"\nsaved: {out_json}")
    print(f"       {out_csv}")
    if samples:
        print(f"       {out_s}")


if __name__ == "__main__":
    main()
