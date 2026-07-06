import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev


METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_weighted",
    "recall_weighted",
    "auc_roc",
    "loss",
    "error_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate seed/fold runs into mean +/- std tables."
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pattern", default="*/run_manifest.json")
    parser.add_argument("--out-csv", default="outputs/aggregate_summary.csv")
    parser.add_argument("--raw-csv", default="outputs/aggregate_runs.csv")
    parser.add_argument("--markdown", default="outputs/aggregate_summary.md")
    parser.add_argument("--include-failed", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_metric_file(exp_dir):
    candidates = [
        os.path.join(exp_dir, "metrics", "test_metrics.json"),
        os.path.join(exp_dir, "metrics", "metrics.json"),
        os.path.join(exp_dir, "evaluation", "metrics.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(exp_dir, "**", "test_metrics.json"), recursive=True)
    if matches:
        return matches[0]
    matches = glob.glob(os.path.join(exp_dir, "**", "metrics.json"), recursive=True)
    return matches[0] if matches else None


def load_run(manifest_path, include_failed=False):
    exp_dir = os.path.dirname(manifest_path)
    manifest = load_json(manifest_path)
    if manifest.get("status") == "failed" and not include_failed:
        return None

    metric_file = find_metric_file(exp_dir)
    if not metric_file:
        return None
    metrics = load_json(metric_file)

    summary_path = os.path.join(exp_dir, "metrics", "training_summary.json")
    summary = load_json(summary_path) if os.path.exists(summary_path) else {}

    row = {
        "experiment": manifest.get("experiment", os.path.basename(exp_dir)),
        "experiment_name": manifest.get("experiment_name", os.path.basename(exp_dir)),
        "seed": manifest.get("seed"),
        "fold": manifest.get("fold"),
        "n_folds": manifest.get("n_folds"),
        "pretrained": manifest.get("pretrained", False),
        "status": manifest.get("status"),
        "metric_file": metric_file,
        "best_val_acc": summary.get("best_val_acc"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "total_epochs_trained": summary.get("total_epochs_trained"),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    row["num_samples"] = metrics.get("num_samples")
    return row


def numeric_values(rows, key):
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(val):
            vals.append(float(val))
    return vals


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row.get("pretrained", False))].append(row)

    summaries = []
    for (experiment, pretrained), group_rows in sorted(groups.items()):
        out = {
            "experiment": experiment,
            "pretrained": pretrained,
            "runs": len(group_rows),
            "seeds": ",".join(str(s) for s in sorted({r["seed"] for r in group_rows})),
            "folds": ",".join(str(f) for f in sorted({r["fold"] for r in group_rows})),
        }
        for key in METRIC_KEYS + ("best_val_acc", "best_val_loss", "best_epoch"):
            vals = numeric_values(group_rows, key)
            if not vals:
                out[f"{key}_mean"] = None
                out[f"{key}_std"] = None
                continue
            out[f"{key}_mean"] = mean(vals)
            out[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summaries.append(out)
    return summaries


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(mean_value, std_value):
    if mean_value is None:
        return "-"
    return f"{mean_value * 100:.2f} +/- {std_value * 100:.2f}"


def write_markdown(path, summaries):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        "| Experiment | Runs | Seeds | Folds | Acc (%) | Macro F1 (%) | AUC (%) | Best Val Acc (%) |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {experiment} | {runs} | {seeds} | {folds} | {acc} | {f1} | {auc} | {val} |".format(
                experiment=row["experiment"],
                runs=row["runs"],
                seeds=row["seeds"],
                folds=row["folds"],
                acc=fmt_pct(row.get("accuracy_mean"), row.get("accuracy_std")),
                f1=fmt_pct(row.get("f1_macro_mean"), row.get("f1_macro_std")),
                auc=fmt_pct(row.get("auc_roc_mean"), row.get("auc_roc_std")),
                val=fmt_pct(row.get("best_val_acc_mean"), row.get("best_val_acc_std")),
            )
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    manifest_paths = glob.glob(os.path.join(args.output_dir, args.pattern))
    rows = []
    for path in manifest_paths:
        row = load_run(path, include_failed=args.include_failed)
        if row:
            rows.append(row)

    summaries = summarize(rows)
    write_csv(args.raw_csv, rows)
    write_csv(args.out_csv, summaries)
    write_markdown(args.markdown, summaries)

    print(f"Loaded runs: {len(rows)}")
    print(f"Wrote raw runs: {args.raw_csv}")
    print(f"Wrote summary: {args.out_csv}")
    print(f"Wrote markdown: {args.markdown}")


if __name__ == "__main__":
    main()
