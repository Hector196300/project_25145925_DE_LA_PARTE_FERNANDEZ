"""
visualise.py

Figures for the report.

    overlays     predicted mask and box drawn on the input image, next to
                 the ground truth, for a handful of val/test images
    confusion    confusion matrix heatmap from a saved metrics file
    curves       training and validation curves from a run's history
    compare      grouped bars comparing several experiments

USAGE
-----
    # qualitative overlays on the smartphone test set
    python src/visualise.py overlays --weights weights/baseline_best.pt \
        --dataset smartphone_dataset/25145925_SURNAME --n 8 \
        --out results/figs/exp2_overlays.png

    # confusion matrix from an evaluation
    python src/visualise.py confusion \
        --metrics results/exp2_phone_zeroshot_metrics.json \
        --out results/figs/exp2_confusion.png

    # training curves
    python src/visualise.py curves --history results/baseline_history.json \
        --out results/figs/baseline_curves.png

    # compare experiments side by side
    python src/visualise.py compare \
        --metrics results/exp1_realsense_test_metrics.json \
                  results/exp2_phone_zeroshot_metrics.json \
                  results/exp3_phone_strongaug_metrics.json \
        --out results/figs/experiment_comparison.png

Figures are saved, never shown, so this runs headless over SSH on the
CS machines without an X display.
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")            # no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import denormalise, load_json


# ----------------------------------------------------------------------

def _save(fig, out: str):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------
# qualitative overlays
# ----------------------------------------------------------------------

def overlays(args):
    import torch
    from dataloader import make_test_dataloader, make_dataloaders, CLASS_NAME
    from model import HandGestureNet, cxcywh_to_xyxy, box_iou
    from evaluate import boxes_from_masks

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.cpu else "cpu")
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    targs = ck.get("args", {})
    width = targs.get("width", 32)
    size = tuple(targs.get("size", [320, 240]))

    model = HandGestureNet(num_classes=len(CLASS_NAME), width=width).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    if args.val_split:
        _, loader = make_dataloaders(root=args.data_root, batch_size=args.n,
                                     size=size,
                                     val_fraction=targs.get("val_fraction", 0.2),
                                     seed=targs.get("seed", 0))
        title = "validation split"
    else:
        loader = make_test_dataloader(args.dataset, batch_size=args.n, size=size)
        title = os.path.basename(args.dataset.rstrip("/\\"))

    batch = next(iter(loader))
    images = batch["image"].to(device)

    with torch.no_grad():
        out = model(images)
        probs = torch.softmax(out["class_logits"], dim=1)
        conf, pred_label = probs.max(dim=1)
        pred_mask = (torch.sigmoid(out["mask_logits"]) > 0.5).float()
        reg_box = cxcywh_to_xyxy(out["box_cxcywh"], size[0], size[1])
        msk_box = boxes_from_masks(pred_mask)

    n = min(args.n, images.shape[0])
    # Full text-width in IEEE two-column is 7.16in, so anything wider than
    # about 4 columns becomes unreadable once scaled down.
    per = 1.75 if args.compact else 2.6
    fig, axes = plt.subplots(2, n, figsize=(per * n, 3.8 if args.compact else 5.6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        img = denormalise(images[i])
        gt_m = batch["mask"][i, 0].numpy()
        pr_m = pred_mask[i, 0].cpu().numpy()
        gt_b = batch["box"][i].numpy()
        rb = reg_box[i].cpu().numpy()
        mb = msk_box[i].cpu().numpy()

        gt_lbl = CLASS_NAME[int(batch["label"][i])]
        pr_lbl = CLASS_NAME[int(pred_label[i])]
        c = float(conf[i])
        iou_r = float(box_iou(reg_box[i:i+1], batch["box"][i:i+1].to(device)))
        iou_m = float(box_iou(msk_box[i:i+1], batch["box"][i:i+1].to(device)))

        # top row: ground truth
        ax = axes[0, i]
        ax.imshow(img)
        ax.imshow(np.ma.masked_where(gt_m < 0.5, gt_m), alpha=0.45, cmap="Greens")
        ax.add_patch(Rectangle((gt_b[0], gt_b[1]), gt_b[2] - gt_b[0],
                               gt_b[3] - gt_b[1], fill=False,
                               edgecolor="lime", lw=1.6))
        ax.set_title(f"GT: {gt_lbl}", fontsize=7 if args.compact else 9)
        ax.axis("off")

        # bottom row: prediction
        ax = axes[1, i]
        ax.imshow(img)
        ax.imshow(np.ma.masked_where(pr_m < 0.5, pr_m), alpha=0.45, cmap="Reds")
        ax.add_patch(Rectangle((rb[0], rb[1]), rb[2] - rb[0], rb[3] - rb[1],
                               fill=False, edgecolor="red", lw=1.4,
                               label="regressed"))
        ax.add_patch(Rectangle((mb[0], mb[1]), mb[2] - mb[0], mb[3] - mb[1],
                               fill=False, edgecolor="orange", lw=1.4,
                               linestyle="--", label="from mask"))
        colour = "black" if pr_lbl == gt_lbl else "crimson"
        ax.set_title(f"{pr_lbl} ({c:.2f})\nIoU {iou_r:.2f}/{iou_m:.2f}",
                     fontsize=6.5 if args.compact else 8, color=colour)
        ax.axis("off")

    axes[0, 0].set_ylabel("ground truth")
    axes[1, 0].set_ylabel("prediction")
    if not args.compact:
        fig.suptitle(f"{title} — top: ground truth,  bottom: prediction "
                     f"(red box regressed, dashed orange derived from mask)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()
    _save(fig, args.out)


# ----------------------------------------------------------------------
# confusion matrix
# ----------------------------------------------------------------------

def confusion(args):
    m = load_json(args.metrics)
    cm = np.array(m["confusion_matrix"])
    names = m["class_names"]

    # row-normalise so classes with different support are comparable
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sums, 1))

    size = (3.6, 3.2) if args.compact else (7.2, 6.2)
    fs = 6 if args.compact else 9
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(names, fontsize=fs)
    ax.set_xlabel("predicted", fontsize=fs + 1)
    ax.set_ylabel("true", fontsize=fs + 1)

    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=fs - 1,
                    color="white" if norm[i, j] > 0.5 else "black")

    acc = m["classification"]["top1_accuracy"]
    f1 = m["classification"]["macro_f1"]
    name = os.path.basename(args.metrics).replace("_metrics.json", "")
    ax.set_title(f"top-1 {acc:.3f}   macro F1 {f1:.3f}" if args.compact
                 else f"{name}\ntop-1 {acc:.3f}   macro F1 {f1:.3f}",
                 fontsize=fs + 2)
    if not args.compact:
        fig.colorbar(im, ax=ax, fraction=0.046,
                     label="proportion of true class")
    _save(fig, args.out)


# ----------------------------------------------------------------------
# training curves
# ----------------------------------------------------------------------

def curves(args):
    h = load_json(args.history)
    ep = [e["epoch"] for e in h]

    if args.compact:
        # Two panels side by side, sized for one IEEE column (3.5in).
        # The per-task loss breakdown is dropped: at column width its
        # four lines are unreadable, and the validation metrics panel
        # carries the same story more directly.
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

        ax = axes[0]
        ax.plot(ep, [e["train"]["total"] for e in h], label="train")
        ax.plot(ep, [e["val"]["loss"] for e in h], label="val")
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("total loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(ep, [e["val"]["cls_acc"] for e in h], label="class acc")
        ax.plot(ep, [e["val"]["mask_iou"] for e in h], label="mask IoU")
        ax.plot(ep, [e["val"]["det_acc@0.5"] for e in h], label="det@0.5")
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("validation metric", fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.tight_layout()
        _save(fig, args.out)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ep, [e["train"]["total"] for e in h], label="train")
    ax.plot(ep, [e["val"]["loss"] for e in h], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss")
    ax.set_title("loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep, [e["train"]["mask_dice"] for e in h], label="mask dice")
    ax.plot(ep, [e["train"]["class"] for e in h], label="class CE")
    ax.plot(ep, [e["train"]["box_giou"] for e in h], label="box GIoU")
    ax.plot(ep, [e["train"]["box_l1"] for e in h], label="box L1")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss component")
    ax.set_title("training loss by task"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep, [e["val"]["cls_acc"] for e in h], label="class acc")
    ax.plot(ep, [e["val"]["mask_iou"] for e in h], label="mask IoU")
    ax.plot(ep, [e["val"]["det_acc@0.5"] for e in h], label="det acc @0.5")
    ax.plot(ep, [e["val"]["box_iou"] for e in h], label="mean box IoU")
    ax.set_xlabel("epoch"); ax.set_ylabel("metric")
    ax.set_ylim(0, 1)
    ax.set_title("validation metrics"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    name = os.path.basename(args.history).replace("_history.json", "")
    fig.suptitle(f"training run: {name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))     # leave room for the suptitle
    _save(fig, args.out)


# ----------------------------------------------------------------------
# experiment comparison
# ----------------------------------------------------------------------

def compare(args):
    data = [load_json(p) for p in args.metrics]
    names = [os.path.basename(p).replace("_metrics.json", "")
             for p in args.metrics]

    rows = [
        ("detection", "accuracy@0.5IoU", "det acc\n@0.5 IoU"),
        ("detection", "mean_box_iou", "mean\nbox IoU"),
        ("segmentation", "mean_iou", "mask\nmean IoU"),
        ("segmentation", "mean_dice", "Dice"),
        ("classification", "top1_accuracy", "top-1\naccuracy"),
        ("classification", "macro_f1", "macro F1"),
    ]

    x = np.arange(len(rows))
    w = 0.8 / len(data)

    fig, ax = plt.subplots(figsize=(7.0, 3.0) if args.compact else (10.5, 4.6))
    for i, (d, n) in enumerate(zip(data, names)):
        vals = [d[g][k] for g, k, _ in rows]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=5.5 if args.compact else 7)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, _, lbl in rows],
                       fontsize=7 if args.compact else 9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.legend(fontsize=6.5 if args.compact else 8)
    ax.grid(axis="y", alpha=0.3)
    if not args.compact:
        ax.set_title("experiment comparison")
    fig.tight_layout()
    _save(fig, args.out)


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Figures for the report")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("overlays", help="mask and box overlays on images")
    o.add_argument("--weights", required=True)
    o.add_argument("--dataset")
    o.add_argument("--val-split", action="store_true")
    o.add_argument("--data-root", default="rgb_only")
    o.add_argument("--n", type=int, default=6)
    o.add_argument("--cpu", action="store_true")
    o.add_argument("--out", default="results/figs/overlays.png")
    o.set_defaults(func=overlays)

    c = sub.add_parser("confusion", help="confusion matrix heatmap")
    c.add_argument("--metrics", required=True)
    c.add_argument("--out", default="results/figs/confusion.png")
    c.set_defaults(func=confusion)

    t = sub.add_parser("curves", help="training and validation curves")
    t.add_argument("--history", required=True)
    t.add_argument("--out", default="results/figs/curves.png")
    t.set_defaults(func=curves)

    k = sub.add_parser("compare", help="bar chart across experiments")
    k.add_argument("--metrics", nargs="+", required=True)
    k.add_argument("--out", default="results/figs/comparison.png")
    k.set_defaults(func=compare)

    for sp in (o, c, t, k):
        sp.add_argument("--compact", action="store_true",
                        help="size the figure for a two-column IEEE paper")

    args = p.parse_args()
    if args.cmd == "overlays" and not args.dataset and not args.val_split:
        p.error("overlays needs either --dataset or --val-split")
    args.func(args)


if __name__ == "__main__":
    main()
