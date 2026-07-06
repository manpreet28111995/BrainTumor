import torch
import torch.nn as nn


class SpectralFusion(nn.Module):
    """
    FFT-based spectral feature modulation.

    This is the core novelty of FLUX-Net.
    Transforms features to frequency domain, applies a learnable
    radial frequency gate, modulates channels, and transforms back.

    The learnable radius controls low-pass vs high-pass emphasis:
    - radius → 1: mostly low-pass (tumor core, homogeneous regions)
    - radius → 0: mostly high-pass (tumor edges, texture boundaries)

    Numerical safety:
    - All FFT ops are forced to float32 to prevent AMP (float16) overflow.
    - nan_to_num guards prevent NaN propagation from extreme activations.
    - Complex magnitudes are clamped before IFFT.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.radius_logit = nn.Parameter(torch.zeros(dim, 1, 1))
        self.slope = nn.Parameter(torch.ones(dim, 1, 1) * 10.0)
        self.channel_mod = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        # Force float32 context to prevent AMP float16 overflow in FFT and Conv2d
        # This makes the entire block mathematically stable
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f32 = x.float()

            # FFT with ortho normalization
            x_fft = torch.fft.fft2(x_f32, norm="ortho")
            x_shifted = torch.fft.fftshift(x_fft)

            # Component-wise nan_to_num on complex tensor (autograd compatible)
            x_shifted_real = torch.nan_to_num(x_shifted.real, nan=0.0, posinf=1e4, neginf=-1e4)
            x_shifted_imag = torch.nan_to_num(x_shifted.imag, nan=0.0, posinf=1e4, neginf=-1e4)
            x_shifted = torch.complex(x_shifted_real, x_shifted_imag)

            # Learnable radial frequency gate
            radius = torch.sigmoid(self.radius_logit)
            fy = torch.linspace(-1, 1, H, device=x.device)
            fx = torch.linspace(-1, 1, W, device=x.device)
            gy, gx = torch.meshgrid(fy, fx, indexing="ij")
            dist = torch.sqrt(gx ** 2 + gy ** 2).unsqueeze(0) / 1.414
            mask = torch.sigmoid(self.slope * (radius - dist))

            # Channel modulation in float32
            x_real = self.channel_mod(x_shifted.real)
            x_imag = self.channel_mod(x_shifted.imag)
            x_mod = torch.complex(x_real, x_imag)

            # Apply frequency gate (soft mask)
            x_gated = x_mod * mask

            # Clamp complex magnitudes to safe range
            magnitude = x_gated.abs().clamp(max=3e4)
            phase = x_gated.angle()
            x_gated = torch.polar(magnitude, phase)

            # IFFT back to spatial domain
            x_out = torch.fft.ifft2(torch.fft.ifftshift(x_gated), norm="ortho").real

            # Clamp output to safe float16 range
            x_out = torch.nan_to_num(x_out, nan=0.0, posinf=3e4, neginf=-3e4)
            x_out = x_out.clamp(-3e4, 3e4)

        return x_out.to(orig_dtype)
