"""
utils.py

Shared helpers used by train.py, evaluate.py and visualise.py.

Kept deliberately small: anything specific to one stage lives in that
stage's module. What is here is either used in more than one place, or
is a pure function worth testing in isolation.
"""

import os
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------

def box_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Tight [x_min, y_min, x_max, y_max] around the non-zero pixels.

    Returns None for an empty mask rather than a degenerate box, so the
    caller decides what an absent hand means.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
    )


def iou_xyxy(a: Sequence[float], b: Sequence[float], eps: float = 1e-7) -> float:
    """IoU between two single boxes in xyxy form."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + eps)


def scale_box(box: Sequence[float], from_size: Tuple[int, int],
              to_size: Tuple[int, int]) -> np.ndarray:
    """
    Rescale a box between two image sizes, each given as (width, height).

    Needed when overlaying predictions made at the model's input size
    onto the original 640x480 frame.
    """
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    return np.array([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
                    dtype=np.float32)


# ----------------------------------------------------------------------
# segmentation metrics
# ----------------------------------------------------------------------

def mask_iou(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    """IoU of two binary masks."""
    p = pred > 0.5
    t = true > 0.5
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter / (union + eps))


def dice(pred: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
    """
    Dice coefficient of two binary masks.

    Dice and IoU are monotonically related, but both are reported because
    the coursework asks for both, and Dice is the more forgiving of the
    two on small objects.
    """
    p = pred > 0.5
    t = true > 0.5
    inter = np.logical_and(p, t).sum()
    return float(2 * inter / (p.sum() + t.sum() + eps))


# ----------------------------------------------------------------------
# classification metrics
# ----------------------------------------------------------------------

def confusion_matrix(true: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray):
    """
    Precision, recall, F1 and support per class, from a confusion matrix.

    A class with no true examples gets an F1 of 0 and is still counted in
    the macro average - the conservative choice, and the one sklearn
    makes by default.
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
        d = precision[i] + recall[i]
        f1[i] = 2 * precision[i] * recall[i] / d if d > 0 else 0.0

    return precision, recall, f1, support


def macro_f1(true: np.ndarray, pred: np.ndarray, n: int) -> float:
    _, _, f1, _ = per_class_prf(confusion_matrix(true, pred, n))
    return float(f1.mean())


# ----------------------------------------------------------------------
# images
# ----------------------------------------------------------------------

# ImageNet statistics, matching dataloader.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalise(tensor) -> np.ndarray:
    """
    Turn a normalised (3, H, W) tensor back into a uint8 RGB image.

    Used for overlays, where the figure has to show what the model
    actually saw, augmentation and all.
    """
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# results files
# ----------------------------------------------------------------------

def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def summarise_runs(paths: Sequence[str]) -> str:
    """
    A compact table comparing several *_metrics.json files.

    Handy for pasting the headline numbers of all experiments into the
    report in one go.
    """
    rows = [
        ("detection", "accuracy@0.5IoU", "det acc @0.5"),
        ("detection", "mean_box_iou", "mean box IoU"),
        ("segmentation", "mean_iou", "mask mean IoU"),
        ("segmentation", "mean_dice", "Dice"),
        ("classification", "top1_accuracy", "top-1 acc"),
        ("classification", "macro_f1", "macro F1"),
    ]

    data = []
    names = []
    for p in paths:
        data.append(load_json(p))
        names.append(os.path.basename(p).replace("_metrics.json", ""))

    width = max(14, max(len(n) for n in names) + 2)
    lines = [f"{'metric':<16}" + "".join(f"{n:>{width}}" for n in names)]
    lines.append("-" * len(lines[0]))
    for group, key, label in rows:
        vals = "".join(f"{d[group][key]:>{width}.4f}" for d in data)
        lines.append(f"{label:<16}{vals}")
    return "\n".join(lines)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    # quick self-check of the pure functions
    m = np.zeros((100, 120), dtype=np.uint8)
    m[20:60, 30:80] = 1
    print("box_from_mask:", box_from_mask(m))
    print("empty mask:   ", box_from_mask(np.zeros((10, 10))))
    print("iou identical:", iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]))
    print("iou disjoint: ", iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]))
    print("iou half:     ", round(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 20]), 4))
    print("scale box:    ", scale_box([10, 10, 50, 50], (320, 240), (640, 480)))
    print("mask iou self:", mask_iou(m, m))
    print("dice self:    ", dice(m, m))
    t = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 0])
    p = np.array([0, 1, 1, 1, 2, 2, 2, 0, 2, 0])
    print("macro f1:     ", round(macro_f1(t, p, 3), 6))
