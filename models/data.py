import os
import cv2
import numpy as np
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class BrainTumorDataset(Dataset):
    """Brain tumor classification dataset for Kaggle-7023 format."""
    CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")

    def __init__(self, root_dir: str, split: str = "train",
                 transform=None, class_names: Tuple[str, ...] = None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.class_names = class_names or self.CLASS_NAMES
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.samples: List[Tuple[str, int]] = []
        self._load()

    def _load(self):
        split_dir = os.path.join(self.root_dir,
                                 "Training" if self.split == "train" else "Testing")
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Not found: {split_dir}")
        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            key = cls_name.lower().replace("_", "").replace(" ", "")
            if key in self.class_to_idx:
                idx = self.class_to_idx[key]
            else:
                matched = False
                for cn in self.class_names:
                    if cn in key or key in cn:
                        idx = self.class_to_idx[cn]
                        matched = True
                        break
                if not matched:
                    continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Failed: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_labels(self):
        return np.array([s[1] for s in self.samples])

    def get_class_distribution(self):
        dist = {n: 0 for n in self.class_names}
        for _, l in self.samples:
            dist[self.class_names[l]] += 1
        return dist


def get_train_transforms(img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        # ── Spatial ────────────────────────────────────────────────────────
        A.Resize(img_size + 16, img_size + 16),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=15,
            border_mode=0, p=0.5),

        # ── Intensity / Medical-imaging specific ───────────────────────────
        A.OneOf([
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),  # MRI contrast enhancement
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
        ], p=0.5),

        # ── Noise / Blur ───────────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),

        # ── Cutout / Dropout ───────────────────────────────────────────────
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(0.0, 0.05),
            hole_width_range=(0.0, 0.05),
            fill=0, p=0.2),

        # ── Normalize ─────────────────────────────────────────────────────
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def _create_transform_subset(ds, indices, tf):
    class TransformSubset(Dataset):
        def __init__(self, ds, indices, tf):
            self.ds = ds
            self.indices = indices
            self.tf = tf
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            path, label = self.ds.samples[self.indices[i]]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.tf(image=img)["image"]
            return img, label
    return TransformSubset(ds, indices, tf)


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    dc = config.data
    tc = config.training
    train_tf = get_train_transforms(dc.img_size, dc.mean, dc.std)
    val_tf = get_val_transforms(dc.img_size, dc.mean, dc.std)
    full = BrainTumorDataset(dc.data_root, "train", class_names=dc.class_names)
    print(f"Dataset: {len(full)} samples, {full.get_class_distribution()}")
    labels = full.get_labels()

    if dc.use_kfold:
        skf = StratifiedKFold(n_splits=tc.n_folds, shuffle=True, random_state=tc.seed)
        splits = list(skf.split(np.zeros(len(labels)), labels))
        train_idx, val_idx = splits[tc.fold]
        print(f"Fold {tc.fold}/{tc.n_folds}: Train={len(train_idx)}, Val={len(val_idx)}")
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=dc.val_split_ratio,
                                     random_state=tc.seed)
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        print(f"Simple split ({1-dc.val_split_ratio:.0%}/{dc.val_split_ratio:.0%}): "
              f"Train={len(train_idx)}, Val={len(val_idx)}")

    train_ds = _create_transform_subset(full, train_idx, train_tf)
    val_ds = _create_transform_subset(full, val_idx, val_tf)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              num_workers=tc.num_workers, pin_memory=tc.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tc.batch_size * 2, shuffle=False,
                            num_workers=tc.num_workers, pin_memory=tc.pin_memory)

    test_loader = None
    try:
        test_ds = BrainTumorDataset(dc.data_root, "test", val_tf, dc.class_names)
        if len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=tc.batch_size * 2,
                                     shuffle=False, num_workers=tc.num_workers,
                                     pin_memory=tc.pin_memory)
            print(f"Test: {len(test_ds)} samples")
    except FileNotFoundError:
        pass
    return train_loader, val_loader, test_loader
