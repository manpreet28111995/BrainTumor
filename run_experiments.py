import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


FLUX_VARIANTS = {
    "fluxnet": ("ablation", "full"),
    "no_spectral_fusion": ("ablation", "no_spectral_fusion"),
    "no_ssm": ("ablation", "no_ssm"),
    "no_msap": ("ablation", "no_msap"),
}


def parse_csv(value, cast=str):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run seed/fold sweeps and write per-run manifests."
    )
    parser.add_argument(
        "--experiments",
        default="fluxnet",
        help=(
            "Comma-separated experiment names. Use fluxnet, no_spectral_fusion, "
            "no_ssm, no_msap, or any baseline accepted by train_baseline.py."
        ),
    )
    parser.add_argument("--seeds", default="42,123,2025")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--data-root", default=os.path.join("data", "kaggle-7023"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--prefix", default="sweep")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def experiment_command(exp_name, seed, fold, args, experiment_name):
    common = [
        "--data-root", args.data_root,
        "--output-dir", args.output_dir,
        "--experiment-name", experiment_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation-steps", str(args.grad_accumulation_steps),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--n-folds", str(args.n_folds),
        "--seed", str(seed),
        "--fold", str(fold),
    ]
    if args.no_amp:
        common.append("--no-amp")

    if exp_name in FLUX_VARIANTS:
        _, ablation = FLUX_VARIANTS[exp_name]
        return [sys.executable, "train_ablation.py", "--ablation", ablation] + common

    cmd = [sys.executable, "train_baseline.py", "--model", exp_name] + common
    if args.pretrained:
        cmd.append("--pretrained")
    return cmd


def write_manifest(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main():
    args = parse_args()
    experiments = parse_csv(args.experiments)
    seeds = parse_csv(args.seeds, int)
    folds = parse_csv(args.folds, int)
    total = len(experiments) * len(seeds) * len(folds)
    run_idx = 0

    for exp_name in experiments:
        for seed in seeds:
            for fold in folds:
                run_idx += 1
                experiment_name = f"{args.prefix}_{exp_name}_seed{seed}_fold{fold}"
                exp_dir = os.path.join(args.output_dir, experiment_name)
                manifest = {
                    "experiment": exp_name,
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "fold": fold,
                    "n_folds": args.n_folds,
                    "pretrained": bool(args.pretrained),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "created_at": utc_now(),
                    "status": "planned",
                }
                cmd = experiment_command(exp_name, seed, fold, args, experiment_name)
                manifest["command"] = cmd

                print(f"[{run_idx}/{total}] {' '.join(cmd)}")
                if args.dry_run:
                    continue

                manifest_path = os.path.join(exp_dir, "run_manifest.json")
                write_manifest(manifest_path, manifest)
                manifest["status"] = "running"
                write_manifest(manifest_path, manifest)
                result = subprocess.run(cmd, cwd=os.getcwd())
                manifest["returncode"] = result.returncode
                manifest["status"] = "complete" if result.returncode == 0 else "failed"
                manifest["finished_at"] = utc_now()
                write_manifest(manifest_path, manifest)

                if result.returncode != 0 and not args.continue_on_error:
                    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
