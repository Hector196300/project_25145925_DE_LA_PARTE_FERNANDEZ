"""
dataloader.py

Dataset and DataLoader logic for COMP0248 CW1 LSA.

Loads both the supplied RealSense datasets and the smartphone test set
through one interface, so training and evaluation code never needs to know
which camera an image came from.

Each sample is a dict:
    image   FloatTensor (3, H, W), normalised
    mask    FloatTensor (1, H, W), 0.0 or 1.0
    box     FloatTensor (4,)  [x_min, y_min, x_max, y_max] in pixels
            of the *resized* image
    label   LongTensor scalar, gesture class index 0-9
    meta    dict with the source paths and original size (for evaluation
            and qualitative overlays)

Only frames that have a mask are indexed - a sample without one cannot
supply a box or a segmentation target.

Splitting is done BY STUDENT, not by image. Frames within a clip are
nearly identical, so an image-level split would leak almost-duplicates
into validation and inflate the score.

USAGE
-----
    from dataloader import GestureDataset, make_dataloaders, GESTURES

    train_dl, val_dl = make_dataloaders("rgb_only", batch_size=16)

    test_ds = GestureDataset(
        "smartphone_dataset/25145925_DE_LA_PARTE_FERNANDEZ",
        students=None, train=False,
    )
"""

import os
import re
import glob
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------

GESTURES = [
    "G01_call", "G02_dislike", "G03_like", "G04_ok", "G05_one",
    "G06_palm", "G07_peace", "G08_rock", "G09_stop", "G10_three",
]
CLASS_INDEX = {g: i for i, g in enumerate(GESTURES)}
CLASS_NAME = [g.split("_", 1)[1] for g in GESTURES]

# ImageNet statistics, for use with a pretrained backbone
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_SIZE = (320, 240)      # (width, height)

# Student folders that must not be used for training.
EXCLUDE_STUDENTS = {
    # no masks at all - nothing to learn from
    "COMP0248_Manuel de Castro Ribeiro Jardim",
    # duplicate of 25150455_Guan; keeping both would double-weight one person
    "25150455_Guan 2",
}

GESTURE_DIR_RE = re.compile(r"^G\d{2}_[a-z]+$", re.IGNORECASE)


# ----------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------

def find_gesture_root(student_dir: str) -> Optional[str]:
    """
    Return the folder that directly contains the Gxx_name folders.

    Most students place them at the top level, but at least one has them
    nested inside an extra folder, so search a couple of levels down
    rather than assuming a fixed depth.
    """
    if not os.path.isdir(student_dir):
        return None

    def has_gestures(d):
        try:
            return any(
                GESTURE_DIR_RE.match(n) and os.path.isdir(os.path.join(d, n))
                for n in os.listdir(d)
            )
        except OSError:
            return False

    if has_gestures(student_dir):
        return student_dir

    for name in sorted(os.listdir(student_dir)):
        sub = os.path.join(student_dir, name)
        if os.path.isdir(sub) and has_gestures(sub):
            return sub

    return None


def list_students(root: str, exclude: Sequence[str] = ()) -> List[str]:
    """Student folder names under `root`, minus any excluded."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No such directory: {root}")
    names = sorted(
        n for n in os.listdir(root)
        if os.path.isdir(os.path.join(root, n))
    )
    return [n for n in names if n not in set(exclude)]


def split_students(
    students: Sequence[str],
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split student names into train and val lists, deterministically."""
    names = sorted(students)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * val_fraction)))
    val = sorted(names[:n_val])
    train = sorted(names[n_val:])
    return train, val


# ----------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------

class GestureDataset(Dataset):
    """
    Indexes every annotated frame under one or more student folders.

    Parameters
    ----------
    root
        Either a folder containing student folders (the RealSense sets),
        or a single dataset folder containing Gxx_name directly (your
        smartphone set).
    students
        Which student folders to use. None means `root` is itself a
        single dataset.
    size
        (width, height) to resize to. Boxes are scaled to match.
    train
        Enables augmentation. Always False for val and test.
    """

    def __init__(
        self,
        root: str,
        students: Optional[Sequence[str]] = None,
        size: Tuple[int, int] = DEFAULT_SIZE,
        train: bool = False,
        augment_strength: float = 1.0,
        domain_aug: bool = False,
    ):
        self.root = root
        self.size = size
        self.train = train
        self.augment_strength = augment_strength
        # When True, _augment_domain replaces _augment: same kinds of
        # transformation, but with ranges chosen to span the measured
        # RealSense-to-smartphone gap rather than generic jitter.
        self.domain_aug = domain_aug

        if students is None:
            roots = [(os.path.basename(root.rstrip("/\\")), find_gesture_root(root))]
        else:
            roots = [(s, find_gesture_root(os.path.join(root, s))) for s in students]

        self.samples = []
        self.skipped_students = []

        for student, gesture_root in roots:
            if gesture_root is None:
                self.skipped_students.append(student)
                continue
            self._index_student(student, gesture_root)

        if not self.samples:
            raise RuntimeError(
                f"No annotated frames found under {root}. "
                f"Check the path and that annotation/ folders are populated."
            )

    def _index_student(self, student: str, gesture_root: str):
        for gesture in sorted(os.listdir(gesture_root)):
            if not GESTURE_DIR_RE.match(gesture):
                continue
            key = gesture.lower()
            # tolerate case differences between students
            match = next((g for g in GESTURES if g.lower() == key), None)
            if match is None:
                continue

            gdir = os.path.join(gesture_root, gesture)
            for clip in sorted(os.listdir(gdir)):
                cdir = os.path.join(gdir, clip)
                ann_dir = os.path.join(cdir, "annotation")
                rgb_dir = os.path.join(cdir, "rgb")
                if not (os.path.isdir(ann_dir) and os.path.isdir(rgb_dir)):
                    continue

                for mask_path in sorted(glob.glob(os.path.join(ann_dir, "*.png"))):
                    frame = os.path.basename(mask_path)
                    rgb_path = os.path.join(rgb_dir, frame)
                    if not os.path.exists(rgb_path):
                        continue          # mask with no matching photo
                    self.samples.append({
                        "student": student,
                        "gesture": match,
                        "clip": clip,
                        "frame": frame,
                        "rgb": rgb_path,
                        "mask": mask_path,
                    })

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]

        image = cv2.imread(rec["rgb"], cv2.IMREAD_COLOR)
        mask = cv2.imread(rec["mask"], cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            # a corrupt file should not kill a training run
            return self.__getitem__((i + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        if mask.shape[:2] != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.uint8)

        if self.train:
            if self.domain_aug:
                image, mask = self._augment_domain(image, mask)
            else:
                image, mask = self._augment(image, mask)

        w, h = self.size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        box = box_from_mask(mask)
        if box is None:
            # augmentation can push the hand out of frame; fall back to
            # the whole image rather than emitting an invalid target
            box = np.array([0, 0, w, h], dtype=np.float32)

        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        return {
            "image": image,
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "box": torch.from_numpy(box).float(),
            "label": torch.tensor(CLASS_INDEX[rec["gesture"]], dtype=torch.long),
            "meta": {
                "rgb": rec["rgb"],
                "mask": rec["mask"],
                "student": rec["student"],
                "gesture": rec["gesture"],
                "clip": rec["clip"],
                "frame": rec["frame"],
                "orig_size": (orig_w, orig_h),
            },
        }

    # ------------------------------------------------------------------

    def _augment(self, image, mask):
        """
        Geometric and photometric augmentation.

        The photometric side is deliberately aggressive: the model is
        trained only on RealSense images but must generalise to phone
        images, so colour, exposure and blur variation is the main lever
        available for Experiment 3.

        NOTE: no horizontal flip. Every image is a right hand, and
        mirroring would produce left hands that never occur at test time.
        """
        s = self.augment_strength
        h, w = image.shape[:2]

        # --- geometric ---
        if random.random() < 0.8 * s:
            angle = random.uniform(-15, 15) * s
            scale = random.uniform(0.85, 1.15)
            tx = random.uniform(-0.08, 0.08) * w * s
            ty = random.uniform(-0.08, 0.08) * h * s
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        img = image.astype(np.float32)

        # --- photometric ---
        if random.random() < 0.8 * s:
            img *= random.uniform(0.6, 1.4)                      # exposure
        if random.random() < 0.6 * s:
            mean = img.mean()
            img = (img - mean) * random.uniform(0.7, 1.3) + mean  # contrast
        if random.random() < 0.6 * s:
            gain = np.array([random.uniform(0.85, 1.15) for _ in range(3)],
                            dtype=np.float32)
            img *= gain                                           # white balance
        if random.random() < 0.3 * s:
            grey = img.mean(axis=2, keepdims=True)
            img = grey + (img - grey) * random.uniform(0.5, 1.5)  # saturation

        img = np.clip(img, 0, 255)

        if random.random() < 0.3 * s:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)                # defocus
        if random.random() < 0.3 * s:
            img += np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.float32)
        if random.random() < 0.2 * s:
            # JPEG-style compression artefacts, common in phone images
            q = random.randint(30, 80)
            ok, enc = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8), mask

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _augment_domain(self, image, mask):
        """
        Domain-targeted augmentation for Experiment 3.

        The generic version above jitters everything mildly. This one is
        shaped by measuring both datasets (see domain_gap.py). Three gaps
        put the smartphone data outside the training distribution:

            hand area      3.19x larger  (4.2 sd above the training mean)
            hand saturation 1.70x higher (outside the 10-90 percentile)
            red channel     1.33x higher (outside the 10-90 percentile)

        So the scale range is widened to cover hands up to roughly twice
        the linear size, and the colour ranges are shifted upward rather
        than centred on 1.0. Everything else is inherited from the
        generic version, since it was already inside range.

        No smartphone image is used here - only the measurement of the
        gap informed the choice of ranges.
        """
        s = self.augment_strength
        h, w = image.shape[:2]

        # --- geometric, scale-targeted ------------------------------
        # Zoom about the hand's centroid rather than the image centre,
        # so that magnifying the image does not push the hand out of
        # frame. Without this, large zoom factors mostly produce
        # unusable samples.
        ys, xs = np.nonzero(mask)
        if len(xs):
            cx, cy = float(xs.mean()), float(ys.mean())
        else:
            cx, cy = w / 2, h / 2

        if random.random() < 0.9:
            angle = random.uniform(-20, 20) * s
            # log-uniform so that shrinking and magnifying are equally
            # likely; the upper end covers the 1.8x linear gap measured
            scale = float(np.exp(random.uniform(np.log(0.7), np.log(2.2))))
            tx = random.uniform(-0.06, 0.06) * w
            ty = random.uniform(-0.06, 0.06) * h

            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            # re-centre the hand, then apply the small random shift
            M[0, 2] += (w / 2 - cx) + tx
            M[1, 2] += (h / 2 - cy) + ty

            new_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
            new_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

            # keep the result only if most of the hand survived; a heavily
            # clipped hand is a bad training target whatever the scale
            before = mask.sum()
            if new_mask.sum() >= 0.5 * before and new_mask.sum() > 20:
                image, mask = new_img, new_mask

        img = image.astype(np.float32)

        # --- photometric, shifted toward the measured test statistics ---
        if random.random() < 0.8:
            img *= random.uniform(0.7, 1.3)
        if random.random() < 0.6:
            mean = img.mean()
            img = (img - mean) * random.uniform(0.7, 1.3) + mean

        # saturation: test hands are 1.7x more saturated, so the range is
        # centred above 1.0 rather than around it
        if random.random() < 0.8:
            grey = img.mean(axis=2, keepdims=True)
            img = grey + (img - grey) * random.uniform(0.8, 2.0)

        # warmth: test frames have a higher red mean
        if random.random() < 0.7:
            gain = np.array([random.uniform(0.85, 1.10),      # B
                             random.uniform(0.90, 1.10),      # G
                             random.uniform(1.00, 1.35)],     # R
                            dtype=np.float32)
            img *= gain

        img = np.clip(img, 0, 255)

        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.3:
            img += np.random.normal(0, random.uniform(2, 10),
                                    img.shape).astype(np.float32)
        if random.random() < 0.25:
            q = random.randint(30, 80)
            ok, enc = cv2.imencode(".jpg", img.clip(0, 255).astype(np.uint8),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8), mask

    # ------------------------------------------------------------------

    def class_counts(self):
        counts = [0] * len(GESTURES)
        for rec in self.samples:
            counts[CLASS_INDEX[rec["gesture"]]] += 1
        return counts

    def summary(self) -> str:
        students = sorted({r["student"] for r in self.samples})
        counts = self.class_counts()
        lines = [
            f"{len(self.samples)} annotated frame(s) "
            f"from {len(students)} student folder(s)",
            "  " + "  ".join(f"{n}:{c}" for n, c in zip(CLASS_NAME, counts)),
        ]
        if self.skipped_students:
            lines.append(f"  skipped (no gesture folders found): "
                         f"{', '.join(self.skipped_students)}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def box_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Tight [x_min, y_min, x_max, y_max] around the non-zero pixels."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array(
        [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32
    )


def collate(batch):
    """Stack tensors, keep meta as a list of dicts."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "box": torch.stack([b["box"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


def make_dataloaders(
    root: str = "rgb_only",
    batch_size: int = 16,
    size: Tuple[int, int] = DEFAULT_SIZE,
    val_fraction: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    augment_strength: float = 1.0,
    domain_aug: bool = False,
):
    """
    Build train and val loaders, split by student.

    num_workers defaults to 0 because multiprocessing workers on Windows
    re-import this module and can be slower than a single process for
    a dataset this size. Raise it on Linux / the CS GPU servers.
    """
    students = list_students(root, exclude=EXCLUDE_STUDENTS)
    train_names, val_names = split_students(students, val_fraction, seed)

    train_ds = GestureDataset(root, train_names, size=size, train=True,
                              augment_strength=augment_strength,
                              domain_aug=domain_aug)
    val_ds = GestureDataset(root, val_names, size=size, train=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, collate_fn=collate,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)

    return train_dl, val_dl


def make_test_dataloader(
    dataset_dir: str,
    batch_size: int = 16,
    size: Tuple[int, int] = DEFAULT_SIZE,
    num_workers: int = 0,
):
    """
    Loader for a single dataset folder - the RealSense test set or your
    smartphone set. No augmentation, no shuffling.
    """
    ds = GestureDataset(dataset_dir, students=None, size=size, train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "rgb_only"

    students = list_students(root, exclude=EXCLUDE_STUDENTS)
    train_names, val_names = split_students(students)
    print(f"{len(students)} usable student folder(s) in '{root}'")
    print(f"  train: {len(train_names)}   val: {len(val_names)}")
    print(f"  val students: {', '.join(val_names)}\n")

    train_ds = GestureDataset(root, train_names, train=True)
    val_ds = GestureDataset(root, val_names, train=False)
    print("TRAIN")
    print(train_ds.summary())
    print("\nVAL")
    print(val_ds.summary())

    sample = train_ds[0]
    print("\nsample tensors:")
    for k in ("image", "mask", "box", "label"):
        v = sample[k]
        print(f"  {k:<6} {tuple(v.shape)}  {v.dtype}")
    print(f"  box    {sample['box'].tolist()}")
    print(f"  label  {sample['label'].item()} "
          f"({CLASS_NAME[sample['label'].item()]})")
