from dataclasses import dataclass, field
from typing import Tuple, Optional
import os


@dataclass
class FLUXConfig:
    # Model selector. Use "fluxnet" for the proposed model, or one of the
    # names returned by models.baselines.available_baselines().
    architecture: str = "fluxnet"
    pretrained: bool = False

    # Architecture
    dims: Tuple[int, ...] = (48, 96, 192, 320)
    depths: Tuple[int, ...] = (2, 3, 6, 2)
    expansion: int = 3
    spectral_fusion: bool = True
    use_ssm: bool = True
    use_msap: bool = True
    freq_radius_init: float = 0.5
    ssm_d_state: int = 8
    ssm_expand: int = 1
    ssm_layers: int = 2
    msap_proj_dim: int = 128
    num_classes: int = 4
    dropout: float = 0.3

    # Stochastic depth: per-stage max drop path rates (increases with depth)
    drop_path_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)
    scheduler: str = "cosine_warm_restarts"
    warmup_epochs: int = 10
    T_0: int = 30
    T_mult: int = 2
    eta_min: float = 1e-6
    epochs: int = 200
    batch_size: int = 16
    grad_accumulation_steps: int = 8
    grad_clip: float = 1.0
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 1e-4
    label_smoothing: float = 0.1
    n_folds: int = 5
    fold: int = 0
    seed: int = 42
    save_best: bool = True
    save_last: bool = True
    save_every_n_epochs: int = 10
    test_every_n_epochs: int = 25
    ema_decay: float = 0.99


@dataclass
class DataConfig:
    data_root: str = ""
    dataset_type: str = "kaggle7023"
    kaggle_train_dir: str = "Training"
    kaggle_test_dir: str = "Testing"
    img_size: int = 224
    class_names: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
    num_classes: int = 4
    mean: Tuple[float, ...] = (0.5, 0.5, 0.5)
    std: Tuple[float, ...] = (0.5, 0.5, 0.5)
    val_split_ratio: float = 0.1
    use_kfold: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "FLUX-Net-BrainTumor"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Tuple[str, ...] = ("brain-tumor", "classification", "flux-net")
    log_model: bool = True
    log_freq: int = 50


@dataclass
class Config:
    model: FLUXConfig = field(default_factory=FLUXConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "outputs"
    experiment_name: str = "flux_v2"

    def __post_init__(self):
        self.refresh_paths()
        if not self.data.data_root:
            parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data.data_root = os.path.join(parent, "data", "kaggle7023")

    def refresh_paths(self):
        self.checkpoint_dir = os.path.join(self.output_dir, self.experiment_name, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, self.experiment_name, "logs")
        self.metrics_dir = os.path.join(self.output_dir, self.experiment_name, "metrics")

    def to_dict(self):
        d = {}
        for section in ["model", "training", "data", "wandb"]:
            sec = getattr(self, section)
            for k, v in sec.__dict__.items():
                d[f"{section}/{k}"] = v
        return d
