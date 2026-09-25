"""Recipient-side PLE adapter. Zero-init the output projection so step 0 is identity."""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - hash/store tests do not need torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _rms_norm(x, weight, eps: float):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight


if nn is not None:

    class PleAdapter(nn.Module):
        def __init__(
            self,
            hidden_size: int = 2560,
            rank: int = 256,
            rms_eps: float = 1e-6,
            gate_bias: float = -3.0,
            arch: str = "bottleneck",
            ple_dim: int | None = None,
        ) -> None:
            super().__init__()
            if arch not in ("bottleneck", "dense", "per_head"):
                raise ValueError(f"unknown adapter arch {arch}")
            self.hidden_size = hidden_size
            self.ple_dim = int(ple_dim) if ple_dim is not None else int(hidden_size)
            self.rank = int(rank)
            self.arch = arch
            self.rms_eps = rms_eps
            self.n_heads = 16
            self.head_dim = self.ple_dim // self.n_heads
            if arch == "per_head" and self.ple_dim != self.n_heads * self.head_dim:
                raise ValueError("per_head arch requires ple_dim divisible by 16")
            if arch != "bottleneck" and self.ple_dim != hidden_size:
                raise ValueError("width mismatch is only supported for bottleneck arch")
            self.ple_norm_weight = nn.Parameter(torch.ones(self.ple_dim))
            self.hidden_norm_weight = nn.Parameter(torch.ones(hidden_size))
            self.w_down = None
            self.w_up = None
            self.w_full = None
            self.heads = None
            self.w_mix = None
            if arch == "dense":
                self.w_full = nn.Linear(hidden_size, hidden_size, bias=False)
            elif arch == "per_head":
                self.heads = nn.ModuleList(
                    [nn.Linear(self.head_dim, self.head_dim, bias=False) for _ in range(self.n_heads)]
                )
                self.w_mix = nn.Linear(hidden_size, hidden_size, bias=False)
            else:
                self.w_down = nn.Linear(self.ple_dim, self.rank, bias=False)
                self.w_up = nn.Linear(self.rank, hidden_size, bias=False)
            self.gate_weight = nn.Parameter(torch.zeros(hidden_size))
            self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias)))
            self.last_gate = None
            self.last_adapter_rms = None
            self.last_hidden_rms = None
            self.reset_parameters()

        def reset_parameters(self) -> None:
            if self.arch == "dense":
                nn.init.zeros_(self.w_full.weight)
            elif self.arch == "per_head":
                for layer in self.heads:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.zeros_(self.w_mix.weight)
            else:
                nn.init.kaiming_uniform_(self.w_down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.w_up.weight)
            nn.init.ones_(self.ple_norm_weight)
            nn.init.ones_(self.hidden_norm_weight)
            nn.init.zeros_(self.gate_weight)

        def extra_trainable_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

        def forward(self, hidden: torch.Tensor, ple_vec: torch.Tensor, enabled: bool = True):
            if not enabled:
                return hidden
            z = _rms_norm(ple_vec, self.ple_norm_weight, self.rms_eps)
            if self.arch == "dense":
                a = self.w_full(z)
            elif self.arch == "per_head":
                e = z.reshape(*z.shape[:-1], self.n_heads, self.head_dim)
                parts = [self.heads[i](e[..., i, :]) for i in range(self.n_heads)]
                a = self.w_mix(torch.cat(parts, dim=-1))
            else:
                a = self.w_up(torch.nn.functional.silu(self.w_down(z)))
            h_n = _rms_norm(hidden, self.hidden_norm_weight, self.rms_eps)
            gate = torch.sigmoid((h_n * self.gate_weight).sum(dim=-1, keepdim=True) + self.gate_bias)
            self.last_gate = gate.detach()
            self.last_adapter_rms = a.detach().float().pow(2).mean().sqrt()
            self.last_hidden_rms = hidden.detach().float().pow(2).mean().sqrt()
            return hidden + gate * a

else:  # pragma: no cover

    class PleAdapter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PleAdapter requires torch")
