import argparse
import os
import random

import numpy as np
import torch

from models import Config, DataConfig, FLUXConfig, Trainer, TrainingConfig


ABLATIONS = ("full", "no_spectral_fusion", "no_ssm", "no_msap")


def parse_args():
    parser = argparse.ArgumentParser(description="Train FLUX-Net ablations.")
    parser.add_argument("--ablation", required=True, choices=ABLATIONS)
    parser.add_argument(
        "--data-root",
        default=os.path.join("data", "kaggle-7023"),
        help="Dataset root containing Training/ and Testing/ directories.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accumulation-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-path", default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def model_config(ablation: str):
    cfg = FLUXConfig()
    if ablation == "no_spectral_fusion":
        cfg.spectral_fusion = False
    elif ablation == "no_ssm":
        cfg.use_ssm = False
    elif ablation == "no_msap":
        cfg.use_msap = False
    return cfg


def build_config(args):
    cfg = Config()
    cfg.model = model_config(args.ablation)
    cfg.training = TrainingConfig(
        optimizer="adamw",
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        scheduler="cosine_warm_restarts",
        warmup_epochs=10,
        T_0=30,
        T_mult=2,
        eta_min=1e-6,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        grad_clip=1.0,
        num_workers=args.num_workers,
        pin_memory=True,
        use_amp=not args.no_amp,
        early_stopping_patience=25,
        early_stopping_min_delta=1e-4,
        label_smoothing=0.1,
        n_folds=args.n_folds,
        fold=args.fold,
        seed=args.seed,
        save_best=True,
        save_last=True,
        save_every_n_epochs=10,
        test_every_n_epochs=25,
        ema_decay=0.99,
    )
    cfg.data = DataConfig(
        data_root=args.data_root,
        dataset_type="kaggle7023",
        kaggle_train_dir="Training",
        kaggle_test_dir="Testing",
        img_size=224,
        class_names=("glioma", "meningioma", "notumor", "pituitary"),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        use_kfold=True,
    )
    cfg.output_dir = args.output_dir
    cfg.experiment_name = args.experiment_name or f"ablation_{args.ablation}"
    cfg.refresh_paths()
    return cfg


def main():
    args = parse_args()
    set_seed(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config(args)
    trainer = Trainer(cfg, device, resume_path=args.resume_path)
    trainer.train()
    if trainer.test_loader is not None:
        best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
        trainer.evaluate(checkpoint_path=best_path if os.path.exists(best_path) else None)


if __name__ == "__main__":
    main()
