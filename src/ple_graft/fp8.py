"""FP8 E4M3FN decode and BF16 scale helpers. Numpy-only so goldens do not need torch."""

from __future__ import annotations

import struct

import numpy as np

# E4M3FN: 1 sign, 4 exp (bias 7), 3 mantissa. exp=15 is NaN (no inf).
_E4M3_BIAS = 7
_LUT = np.empty(256, dtype=np.float32)
for _u in range(256):
    _sign = (_u >> 7) & 1
    _exp = (_u >> 3) & 0xF
    _mant = _u & 0x7
    if _exp == 0:
        _val = (_mant / 8.0) * (2.0 ** (1 - _E4M3_BIAS))
    elif _exp == 0xF:
        _val = float("nan")
    else:
        _val = (1.0 + _mant / 8.0) * (2.0 ** (_exp - _E4M3_BIAS))
    _LUT[_u] = np.float32(-_val if _sign else _val)


def e4m3fn_to_f32(u8: np.ndarray) -> np.ndarray:
    """Decode IEEE-style float8_e4m3fn bytes to float32 (pre-scale)."""
    return _LUT[np.asarray(u8, dtype=np.uint8)]


def bf16_bytes_to_f32(raw: bytes) -> float:
    if len(raw) < 2:
        raise ValueError(f"need 2 bytes of bf16, got {len(raw)}")
    u16 = struct.unpack_from("<H", raw)[0]
    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]


def f32_to_bf16_u16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return (bits >> 16) & 0xFFFF
