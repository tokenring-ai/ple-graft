"""Tiny LoRA wrappers. B is zero-init so the wrapped Linear is identity at step 0."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
            super().__init__()
            if rank < 1:
                raise ValueError("LoRA rank must be >= 1")
            self.base = base
            self.rank = int(rank)
            self.alpha = float(alpha)
            self.scale = self.alpha / self.rank
            in_f = base.in_features
            out_f = base.out_features
            self.lora_A = nn.Parameter(torch.empty(self.rank, in_f, dtype=base.weight.dtype, device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank, dtype=base.weight.dtype, device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            base.weight.requires_grad_(False)
            if base.bias is not None:
                base.bias.requires_grad_(False)

        def forward(self, x):
            return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scale

    def apply_lora(module: nn.Module, rank: int = 16, alpha: float = 16.0) -> int:
        """Replace nn.Linear children with LoRALinear. Returns number of wrapped layers."""
        n = 0
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
                n += 1
            else:
                n += apply_lora(child, rank=rank, alpha=alpha)
        return n

    def lora_state_dict(module: nn.Module) -> dict:
        return {k: v.detach().cpu() for k, v in module.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(module: nn.Module, state: dict) -> None:
        missing = []
        own = module.state_dict()
        for k, v in state.items():
            if k not in own:
                missing.append(k)
                continue
            own[k].copy_(v.to(device=own[k].device, dtype=own[k].dtype))
        if missing:
            raise KeyError(f"LoRA keys not in model: {missing[:8]}")

else:  # pragma: no cover

    class LoRALinear:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("LoRALinear requires torch")

    def apply_lora(*args, **kwargs):
        raise ImportError("apply_lora requires torch")
