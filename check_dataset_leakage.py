import argparse
import csv
import hashlib
import os
from collections import defaultdict

from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check exact and near-duplicate leakage between train/test."
    )
    parser.add_argument("--data-root", default=os.path.join("data", "kaggle-7023"))
    parser.add_argument("--train-dir", default="Training")
    parser.add_argument("--test-dir", default="Testing")
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--max-hamming", type=int, default=2)
    parser.add_argument("--out-csv", default="outputs/leakage_matches.csv")
    return parser.parse_args()


def iter_images(root):
    for cls_name in sorted(os.listdir(root)):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(IMG_EXTS):
                yield os.path.join(cls_dir, fname), cls_name


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path, hash_size):
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size, hash_size))
        pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return bits


def hamming(a, b):
    return (a ^ b).bit_count()


def build_index(root, hash_size):
    exact = defaultdict(list)
    perceptual = []
    for path, cls_name in iter_images(root):
        exact[sha256_file(path)].append((path, cls_name))
        perceptual.append((average_hash(path, hash_size), path, cls_name))
    return exact, perceptual


def main():
    args = parse_args()
    train_root = os.path.join(args.data_root, args.train_dir)
    test_root = os.path.join(args.data_root, args.test_dir)
    train_exact, train_phash = build_index(train_root, args.hash_size)
    rows = []

    for test_path, test_cls in iter_images(test_root):
        digest = sha256_file(test_path)
        for train_path, train_cls in train_exact.get(digest, []):
            rows.append({
                "match_type": "exact_sha256",
                "hamming": 0,
                "test_class": test_cls,
                "train_class": train_cls,
                "test_path": test_path,
                "train_path": train_path,
            })

        test_hash = average_hash(test_path, args.hash_size)
        for train_hash, train_path, train_cls in train_phash:
            dist = hamming(test_hash, train_hash)
            if dist <= args.max_hamming:
                rows.append({
                    "match_type": "near_ahash",
                    "hamming": dist,
                    "test_class": test_cls,
                    "train_class": train_cls,
                    "test_path": test_path,
                    "train_path": train_path,
                })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = [
            "match_type", "hamming", "test_class", "train_class",
            "test_path", "train_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exact_count = sum(1 for row in rows if row["match_type"] == "exact_sha256")
    near_count = sum(1 for row in rows if row["match_type"] == "near_ahash")
    print(f"Exact matches: {exact_count}")
    print(f"Near matches (aHash <= {args.max_hamming}): {near_count}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
