"""
model.py

Multitask hand gesture model for COMP0248 CW1 LSA.

One shared encoder feeds three heads:

    encoder  ->  mask decoder    pixel-wise hand mask
             ->  box head        one bounding box
             ->  class head      10 gesture logits + confidence

Every image in this dataset contains exactly one right hand, so the
detection problem is a single-box regression. That removes the need for
anchors, region proposals and non-maximum suppression, which is why a
compact custom network is appropriate here rather than a detector
framework.

The encoder is written from scratch (no pretrained weights, no
torchvision detection models) to comply with the coursework's
restriction on high-level frameworks.

USAGE
-----
    from model import HandGestureNet, MultitaskLoss

    model = HandGestureNet(num_classes=10)
    out = model(images)                    # dict of predictions
    loss, parts = criterion(out, targets)  # scalar + per-task breakdown
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate the encoder skip, then convolve."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # sizes can differ by a pixel when an input dimension is odd
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

class HandGestureNet(nn.Module):
    """
    Parameters
    ----------
    num_classes
        Number of gesture classes.
    width
        Channel width of the first encoder stage. Later stages double it.
        Halve this if memory is tight; double it if the model underfits.
    dropout
        Applied before the classification head only.
    """

    def __init__(self, num_classes: int = 10, width: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        w = width
        chs = [w, w * 2, w * 4, w * 8, w * 8]

        # --- encoder -------------------------------------------------
        self.enc1 = ConvBlock(3, chs[0])           # full resolution
        self.enc2 = ConvBlock(chs[0], chs[1])      # 1/2
        self.enc3 = ConvBlock(chs[1], chs[2])      # 1/4
        self.enc4 = ConvBlock(chs[2], chs[3])      # 1/8
        self.enc5 = ConvBlock(chs[3], chs[4])      # 1/16
        self.pool = nn.MaxPool2d(2)

        # --- mask decoder --------------------------------------------
        self.up4 = UpBlock(chs[4], chs[3], chs[3])
        self.up3 = UpBlock(chs[3], chs[2], chs[2])
        self.up2 = UpBlock(chs[2], chs[1], chs[1])
        self.up1 = UpBlock(chs[1], chs[0], chs[0])
        self.mask_out = nn.Conv2d(chs[0], 1, 1)

        # --- shared pooled feature for the two vector heads -----------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- classification head -------------------------------------
        # The hand covers only 5-16% of a frame, so a plain global average
        # over the deepest feature map is dominated by background and the
        # gesture signal is diluted. Two changes address this:
        #
        #   1. Mask-guided pooling. The predicted mask weights the average,
        #      so the pooled vector describes the hand rather than the room.
        #      This is the segmentation head helping the classification head,
        #      which is the point of training them jointly.
        #
        #   2. Features from two depths. At stride 16 a hand spans only a few
        #      cells, too coarse to separate "three" from "peace". Stride 8
        #      keeps the finger-level detail those classes need.
        #
        # The unweighted global average of the deepest map is concatenated as
        # well, so the head still has a usable input early in training while
        # the mask is poor.
        cls_in = chs[3] + chs[4] + chs[4]        # masked e4, masked e5, global e5
        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cls_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # --- box head ------------------------------------------------
        # Predicts centre-x, centre-y, width, height, each in [0, 1] via
        # a sigmoid. Parameterising it this way makes every prediction a
        # valid box: x_max > x_min and y_max > y_min hold by construction,
        # which direct xyxy regression does not guarantee.
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chs[4], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    @staticmethod
    def _masked_pool(feat: torch.Tensor, weight: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
        """
        Spatial average of `feat`, weighted by `weight`.

        weight is resized to the feature map's resolution first. A small
        constant is added so that an all-zero mask degrades to a plain
        average rather than dividing by zero.
        """
        w = F.interpolate(weight, size=feat.shape[-2:],
                          mode="bilinear", align_corners=False)
        w = w + eps
        return (feat * w).sum(dim=(2, 3)) / w.sum(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with
            mask_logits  (B, 1, H, W)   pre-sigmoid
            class_logits (B, num_classes)
            box_cxcywh   (B, 4)  normalised to [0, 1]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        mask_logits = self.mask_out(d1)

        # The mask is detached before being used as a pooling weight, so
        # the classification loss cannot reshape the segmentation output
        # to suit itself. The mask head is trained by its own loss alone.
        attn = torch.sigmoid(mask_logits.detach())

        pooled_global = self.gap(e5).flatten(1)
        pooled_e5 = self._masked_pool(e5, attn)
        pooled_e4 = self._masked_pool(e4, attn)

        class_logits = self.cls_head(
            torch.cat([pooled_e4, pooled_e5, pooled_global], dim=1)
        )
        box = torch.sigmoid(self.box_head(self.gap(e5)))

        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "box_cxcywh": box,
        }

    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inference outputs in the form the coursework asks for:
        box in pixels (xyxy), binary mask, class index, confidence.
        """
        out = self.forward(x)
        h, w = x.shape[-2:]

        probs = F.softmax(out["class_logits"], dim=1)
        conf, label = probs.max(dim=1)

        return {
            "box_xyxy": cxcywh_to_xyxy(out["box_cxcywh"], w, h),
            "mask": (torch.sigmoid(out["mask_logits"]) > 0.5).float(),
            "mask_prob": torch.sigmoid(out["mask_logits"]),
            "label": label,
            "confidence": conf,
            "class_probs": probs,
        }


# ----------------------------------------------------------------------
# box conversions
# ----------------------------------------------------------------------

def cxcywh_to_xyxy(box: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Normalised centre-form boxes to absolute [x_min, y_min, x_max, y_max]."""
    cx, cy, bw, bh = box.unbind(-1)
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(box: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Absolute xyxy boxes to normalised centre form."""
    x1, y1, x2, y2 = box.unbind(-1)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return torch.stack([cx, cy, bw, bh], dim=-1)


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Element-wise IoU between two sets of xyxy boxes. Shape (N,)."""
    x1 = torch.max(a[:, 0], b[:, 0])
    y1 = torch.max(a[:, 1], b[:, 1])
    x2 = torch.min(a[:, 2], b[:, 2])
    y2 = torch.min(a[:, 3], b[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + eps)


def generalized_box_iou(a: torch.Tensor, b: torch.Tensor,
                        eps: float = 1e-7) -> torch.Tensor:
    """
    GIoU, which unlike plain IoU still gives a gradient when two boxes
    do not overlap at all - important early in training when predictions
    are far from the target.
    """
    iou = box_iou(a, b, eps)

    x1 = torch.min(a[:, 0], b[:, 0])
    y1 = torch.min(a[:, 1], b[:, 1])
    x2 = torch.max(a[:, 2], b[:, 2])
    y2 = torch.max(a[:, 3], b[:, 3])
    enclosing = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    inter = iou * (area_a + area_b) / (1 + iou + eps)
    union = area_a + area_b - inter

    return iou - (enclosing - union) / (enclosing + eps)


# ----------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice on the sigmoid probabilities.

    Paired with BCE because BCE alone is dominated by the background:
    the hand occupies only 5-16% of a frame in this dataset, so a model
    that predicts all-background already scores well on BCE.
    """
    probs = torch.sigmoid(logits)
    b = probs.shape[0]
    p = probs.reshape(b, -1)
    t = target.reshape(b, -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


class MultitaskLoss(nn.Module):
    """
    Weighted sum of the three task losses.

    The weights matter: the segmentation loss is O(1) per pixel while
    the box loss is O(1) per image, so without balancing, one task
    dominates. The defaults were chosen so all three contribute
    comparable gradient magnitude at initialisation.
    """

    def __init__(
        self,
        w_mask: float = 1.0,
        w_box: float = 5.0,
        w_giou: float = 2.0,
        w_class: float = 1.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.w_mask = w_mask
        self.w_box = w_box
        self.w_giou = w_giou
        self.w_class = w_class
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        outputs
            Whatever HandGestureNet.forward returned.
        targets
            dict with 'mask' (B,1,H,W), 'box' (B,4) in pixels xyxy,
            and 'label' (B,).
        image_size
            (width, height) of the model input, for box normalisation.
        """
        w, h = image_size

        # --- segmentation ---
        mask_bce = self.bce(outputs["mask_logits"], targets["mask"])
        mask_dice = dice_loss(outputs["mask_logits"], targets["mask"])
        loss_mask = mask_bce + mask_dice

        # --- classification ---
        loss_class = self.ce(outputs["class_logits"], targets["label"])

        # --- detection ---
        target_cxcywh = xyxy_to_cxcywh(targets["box"], w, h)
        loss_l1 = F.l1_loss(outputs["box_cxcywh"], target_cxcywh)

        pred_xyxy = cxcywh_to_xyxy(outputs["box_cxcywh"], w, h)
        loss_giou = (1 - generalized_box_iou(pred_xyxy, targets["box"])).mean()

        total = (
            self.w_mask * loss_mask
            + self.w_class * loss_class
            + self.w_box * loss_l1
            + self.w_giou * loss_giou
        )

        parts = {
            "total": float(total.detach()),
            "mask_bce": float(mask_bce.detach()),
            "mask_dice": float(mask_dice.detach()),
            "class": float(loss_class.detach()),
            "box_l1": float(loss_l1.detach()),
            "box_giou": float(loss_giou.detach()),
        }
        return total, parts


# ----------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)

    model = HandGestureNet(num_classes=10, width=32)
    criterion = MultitaskLoss()

    b, w, h = 4, 320, 240
    x = torch.randn(b, 3, h, w)

    out = model(x)
    print(f"parameters: {count_parameters(model):,}")
    print("forward outputs:")
    for k, v in out.items():
        print(f"  {k:<13} {tuple(v.shape)}")

    targets = {
        "mask": (torch.rand(b, 1, h, w) > 0.8).float(),
        "box": torch.tensor([[50., 40., 200., 190.]] * b),
        "label": torch.randint(0, 10, (b,)),
    }
    loss, parts = criterion(out, targets, (w, h))
    print("\nloss parts:")
    for k, v in parts.items():
        print(f"  {k:<10} {v:.4f}")

    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\nbackward ok, {grads} parameter tensors have gradients")

    pred = model.predict(x)
    print("\npredict outputs:")
    for k, v in pred.items():
        print(f"  {k:<12} {tuple(v.shape)}")
    print(f"  first box   {pred['box_xyxy'][0].tolist()}")
    print(f"  confidence  {pred['confidence'][0]:.3f}")
