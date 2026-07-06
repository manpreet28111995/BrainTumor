import argparse
import csv
import os
import resource
import time
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

from models.baselines import available_baselines, build_baseline_model
from models.config import FLUXConfig
from models.flux_net import FLUXNet


MODEL_CHOICES = (
    "fluxnet",
    "fluxnet_no_spectral_fusion",
    "fluxnet_no_ssm",
    "fluxnet_no_msap",
) + available_baselines()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile parameters, FLOPs, latency, and memory."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="fluxnet")
    parser.add_argument("--all", action="store_true", help="Profile all supported models.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    return parser.parse_args()


def build_model(name: str, num_classes: int, pretrained: bool):
    if name.startswith("fluxnet"):
        cfg = FLUXConfig(num_classes=num_classes)
        if name == "fluxnet_no_spectral_fusion":
            cfg.spectral_fusion = False
        elif name == "fluxnet_no_ssm":
            cfg.use_ssm = False
        elif name == "fluxnet_no_msap":
            cfg.use_msap = False
        return FLUXNet(cfg)
    return build_baseline_model(name, num_classes=num_classes, pretrained=pretrained)


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _conv_flops(module: nn.Conv2d, inputs, output):
    batch = output.shape[0]
    out_h, out_w = output.shape[-2:]
    out_channels = module.out_channels
    kernel_ops = module.kernel_size[0] * module.kernel_size[1]
    in_channels = module.in_channels // module.groups
    return batch * out_h * out_w * out_channels * in_channels * kernel_ops


def _linear_flops(module: nn.Linear, inputs, output):
    return inputs[0].shape[0] * module.in_features * module.out_features


def _norm_flops(module, inputs, output):
    return output.numel() * 2


def estimate_flops(model: nn.Module, sample: torch.Tensor) -> int:
    """Hook-based multiply-add estimate; reports FLOPs as 2 * MACs.

    FFT, softmax, activations, pooling, and the Python SSM scan are not fully
    counted. Use this as a consistent architecture-comparison estimate, not a
    hardware profiler replacement.
    """
    macs = 0
    handles = []

    def add_hook(fn):
        def hook(module, inputs, output):
            nonlocal macs
            if isinstance(output, dict):
                return
            macs += int(fn(module, inputs, output))
        return hook

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(add_hook(_conv_flops)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(add_hook(_linear_flops)))
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            handles.append(module.register_forward_hook(add_hook(_norm_flops)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(sample)
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return macs * 2


def measure_latency(model: nn.Module, sample: torch.Tensor, warmup: int, runs: int):
    device = sample.device
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(runs):
                model(sample)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            for _ in range(runs):
                model(sample)
            total_ms = (time.perf_counter() - t0) * 1000.0
    return total_ms / max(runs, 1)


def measure_peak_memory(model: nn.Module, sample: torch.Tensor):
    device = sample.device
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(sample)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with torch.no_grad():
        model(sample)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024.0 if os.uname().sysname == "Darwin" else 1.0
    return max(after - before, 0) / scale


def profile_one(name: str, args, device: torch.device) -> Dict[str, float]:
    model = build_model(name, args.num_classes, args.pretrained).to(device)
    sample = torch.randn(
        args.batch_size, 3, args.img_size, args.img_size, device=device
    )
    total_params, trainable_params = count_params(model)
    flops = estimate_flops(model, sample)
    latency_ms = measure_latency(model, sample, args.warmup, args.runs)
    memory_mb = measure_peak_memory(model, sample)
    return {
        "model": name,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "gflops": flops / 1e9,
        "latency_ms": latency_ms,
        "peak_memory_mb": memory_mb,
        "device": str(device),
    }


def print_table(rows: Iterable[Dict[str, float]]):
    rows = list(rows)
    headers = ["model", "params", "gflops", "latency_ms", "peak_memory_mb", "device"]
    print("\t".join(headers))
    for row in rows:
        print(
            f"{row['model']}\t"
            f"{row['params'] / 1e6:.2f}M\t"
            f"{row['gflops']:.3f}\t"
            f"{row['latency_ms']:.2f}\t"
            f"{row['peak_memory_mb']:.1f}\t"
            f"{row['device']}"
        )


def write_csv(path: str, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = MODEL_CHOICES if args.all else (args.model,)
    rows = [profile_one(name, args, device) for name in names]
    print_table(rows)
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
