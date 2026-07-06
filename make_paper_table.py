import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge aggregate metrics with profiling into paper-ready tables."
    )
    parser.add_argument("--aggregate", default="outputs/aggregate_summary.csv")
    parser.add_argument("--profile", default="outputs/profiling_results.csv")
    parser.add_argument("--out-csv", default="outputs/paper_table.csv")
    parser.add_argument("--out-md", default="outputs/paper_table.md")
    return parser.parse_args()


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_mean_std(row, key, scale=100.0, digits=2):
    mean = as_float(row, f"{key}_mean")
    std = as_float(row, f"{key}_std")
    if mean is None:
        return ""
    std = 0.0 if std is None else std
    return f"{mean * scale:.{digits}f} +/- {std * scale:.{digits}f}"


def fmt_float(value, digits=2):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def fmt_params(value):
    if value in (None, ""):
        return ""
    return f"{float(value) / 1e6:.2f}M"


def index_profile(rows):
    return {row.get("model"): row for row in rows}


PROFILE_ALIASES = {
    "fluxnet": "fluxnet",
    "full": "fluxnet",
    "no_spectral_fusion": "fluxnet_no_spectral_fusion",
    "no_ssm": "fluxnet_no_ssm",
    "no_msap": "fluxnet_no_msap",
}


def build_rows(aggregate_rows, profile_rows):
    profiles = index_profile(profile_rows)
    rows = []
    for metric in aggregate_rows:
        name = metric["experiment"]
        profile = profiles.get(name) or profiles.get(PROFILE_ALIASES.get(name), {})
        rows.append({
            "model": name,
            "pretrained": metric.get("pretrained", "False"),
            "runs": metric.get("runs", ""),
            "seeds": metric.get("seeds", ""),
            "folds": metric.get("folds", ""),
            "params": fmt_params(profile.get("params")),
            "gflops": fmt_float(profile.get("gflops"), 3),
            "latency_ms": fmt_float(profile.get("latency_ms"), 2),
            "peak_memory_mb": fmt_float(profile.get("peak_memory_mb"), 1),
            "accuracy_mean_std": fmt_mean_std(metric, "accuracy"),
            "macro_f1_mean_std": fmt_mean_std(metric, "f1_macro"),
            "auc_roc_mean_std": fmt_mean_std(metric, "auc_roc"),
            "best_val_acc_mean_std": fmt_mean_std(metric, "best_val_acc"),
        })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    headers = [
        "Model", "Runs", "Params", "GFLOPs", "Latency (ms)", "Memory (MB)",
        "Accuracy (%)", "Macro F1 (%)", "AUC (%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {runs} | {params} | {gflops} | {latency} | {memory} | "
            "{acc} | {f1} | {auc} |".format(
                model=row["model"],
                runs=row["runs"],
                params=row["params"],
                gflops=row["gflops"],
                latency=row["latency_ms"],
                memory=row["peak_memory_mb"],
                acc=row["accuracy_mean_std"],
                f1=row["macro_f1_mean_std"],
                auc=row["auc_roc_mean_std"],
            )
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = build_rows(read_csv(args.aggregate), read_csv(args.profile))
    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
