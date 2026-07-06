import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """Single-direction selective state-space model (S6) in pure PyTorch.

    Numerical safety:
    - delta is clamped before exp() to prevent float16/float32 overflow.
    - scan output is guarded with nan_to_num.
    - A_log is initialized with safe negative values.
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand: int = 1):
        super().__init__()
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        # Initialize dt_proj bias to prevent large initial deltas
        nn.init.constant_(self.dt_proj.bias, math.log(0.01))

    def _ssm_scan(self, u, delta, A, B, C):
        """
        Numerically-stable sequential SSM scan in float32.
        - delta is clamped to [-8, 8] before exp to prevent overflow.
        - Intermediate state x is clamped at each step to prevent exponential explosion.
        - Output is clamped and nan_to_num guarded.
        """
        batch, seq, D = u.shape
        N = A.shape[1]

        # Force float32 scan computation
        u_f32 = u.float()
        delta_f32 = delta.float()
        A_f32 = A.float()
        B_f32 = B.float()
        C_f32 = C.float()

        delta_clamped = delta_f32.clamp(-8.0, 8.0)
        deltaA = torch.exp(delta_clamped.unsqueeze(-1) * A_f32.unsqueeze(0).unsqueeze(0))
        deltaB_u = (delta_clamped.unsqueeze(-1) * B_f32.unsqueeze(2)) * u_f32.unsqueeze(-1)

        x = torch.zeros(batch, D, N, device=u.device, dtype=torch.float32)
        ys = []
        for i in range(seq):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            # Hard wall: clamp intermediate state to prevent exponential growth
            x = x.clamp(-1e4, 1e4)
            y = (x * C_f32[:, i].unsqueeze(1)).sum(dim=-1)
            ys.append(y)

        out = torch.stack(ys, dim=1)
        out = torch.nan_to_num(out, nan=0.0, posinf=3e4, neginf=-3e4)
        return out.clamp(-3e4, 3e4)

    def forward(self, x):
        residual = x
        orig_dtype = x.dtype

        # Disable autocast to run full SSM in float32 (standard for stability in recurrent nets)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f32 = x.float()

            x_norm = self.norm(x_f32)
            xz = self.in_proj(x_norm)
            x_main, z = xz.chunk(2, dim=-1)
            x_main = x_main.transpose(1, 2)
            x_main = self.conv1d(x_main)[:, :, :x.shape[1]]
            x_main = x_main.transpose(1, 2)
            x_main = F.silu(x_main)

            A = -torch.exp(self.A_log.clamp(-8.0, 8.0))
            x_dbl = self.x_proj(x_main)
            B, C = x_dbl.chunk(2, dim=-1)
            delta = F.softplus(self.dt_proj(x_main))

            # Clamp inputs to scan to safe boundaries
            B = B.clamp(-100.0, 100.0)
            C = C.clamp(-100.0, 100.0)
            x_main = x_main.clamp(-100.0, 100.0)

            y = self._ssm_scan(x_main, delta, A, B, C)
            y = y + self.D.unsqueeze(0).unsqueeze(0) * x_main
            y = y * F.silu(z)
            out = self.out_proj(y) + residual.float()

            out = torch.nan_to_num(out, nan=0.0, posinf=3e4, neginf=-3e4)
            out = out.clamp(-3e4, 3e4)

        return out.to(orig_dtype)


class BidirectionalS6(nn.Module):
    """Bidirectional S6 SSM block."""

    def __init__(self, d_model, d_state=8, d_conv=4, expand=1):
        super().__init__()
        self.fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.rev = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        y_fwd = self.fwd(x)
        x_rev = torch.flip(x, dims=[1])
        y_rev = self.rev(x_rev)
        y_rev = torch.flip(y_rev, dims=[1])
        merged = torch.cat([y_fwd, y_rev], dim=-1)
        return self.norm(self.merge(merged) + x)


class S6LiteBottleneck(nn.Module):
    """
    S6-Lite SSM Bottleneck.
    Flattens spatial features to tokens, applies bidirectional SSM, reshapes back.
    """

    def __init__(self, d_model=320, d_state=8, d_conv=4, expand=1, num_layers=1):
        super().__init__()
        self.expand = expand
        self.layers = nn.ModuleList([
            BidirectionalS6(d_model, d_state, d_conv, expand) for _ in range(num_layers)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, 49, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        tokens = x.flatten(2).transpose(1, 2)

        pos_embed = self.pos_embed
        if N != pos_embed.shape[1]:
            # Interpolate positional embeddings dynamically (bicubic)
            L = pos_embed.shape[1]
            H_orig = int(math.sqrt(L))
            W_orig = H_orig
            pos_grid = pos_embed.transpose(1, 2).reshape(1, C, H_orig, W_orig)
            pos_grid = F.interpolate(pos_grid, size=(H, W), mode='bicubic', align_corners=False)
            pos_embed = pos_grid.flatten(2).transpose(1, 2)

        tokens = tokens + pos_embed[:, :N, :C]
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)
