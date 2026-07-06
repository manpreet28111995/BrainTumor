import argparse
import csv
import os

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% confidence intervals from predictions.csv."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_predictions(path):
    true, pred = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true.append(row["true_class"])
            pred.append(row["predicted_class"])
    return np.array(true), np.array(pred)


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def bootstrap_ci(y_true, y_pred, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = compute_metrics(y_true, y_pred)
    samples = {key: [] for key in observed}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred[idx])
        for key, value in m.items():
            samples[key].append(value)

    rows = []
    for key, value in observed.items():
        vals = np.array(samples[key], dtype=float)
        rows.append({
            "metric": key,
            "value": value,
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "n_samples": n,
            "n_bootstrap": n_bootstrap,
        })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    y_true, y_pred = read_predictions(args.predictions)
    rows = bootstrap_ci(y_true, y_pred, args.n_bootstrap, args.seed)
    for row in rows:
        print(
            f"{row['metric']}: {row['value'] * 100:.2f}% "
            f"[{row['ci_low'] * 100:.2f}, {row['ci_high'] * 100:.2f}]"
        )
    if args.out_csv:
        write_csv(args.out_csv, rows)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
