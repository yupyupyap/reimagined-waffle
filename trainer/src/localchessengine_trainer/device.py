from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
import sys
from typing import Literal

import torch

DeviceRequest = Literal["auto", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    device_type: str
    torch_version: str
    mps_built: bool
    mps_available: bool
    fallback_used: bool
    reason: str
    python_version: str
    platform: str
    pytorch_enable_mps_fallback: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device_type)


def _mps_built() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built())


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def _probe_device(device_type: str) -> None:
    device = torch.device(device_type)
    value = (torch.ones((2, 2), device=device) * 2).sum()
    observed = float(value.detach().cpu().item())
    if observed != 8.0:
        raise RuntimeError(f"{device_type} probe produced {observed}, expected 8.0")


def _base_selection(
    requested: str,
    device_type: str,
    fallback_used: bool,
    reason: str,
) -> DeviceSelection:
    return DeviceSelection(
        requested=requested,
        device_type=device_type,
        torch_version=torch.__version__,
        mps_built=_mps_built(),
        mps_available=_mps_available(),
        fallback_used=fallback_used,
        reason=reason,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        pytorch_enable_mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    )


def select_device(requested: str = "auto", *, probe: bool = True) -> DeviceSelection:
    normalized = requested.lower()
    if normalized not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cpu")

    if normalized == "cpu":
        if probe:
            _probe_device("cpu")
        return _base_selection("cpu", "cpu", False, "CPU explicitly requested")

    mps_built = _mps_built()
    mps_available = _mps_available()
    if mps_built and mps_available:
        try:
            if probe:
                _probe_device("mps")
            return _base_selection(normalized, "mps", False, "MPS available")
        except RuntimeError as exc:
            if normalized == "mps":
                raise RuntimeError(f"MPS requested but probe failed: {exc}") from exc
            return _base_selection("auto", "cpu", True, f"MPS probe failed: {exc}")

    reason = "PyTorch was not built with MPS support" if not mps_built else "MPS is not available"
    if normalized == "mps":
        raise RuntimeError(f"MPS requested but unavailable: {reason}")

    if probe:
        _probe_device("cpu")
    return _base_selection("auto", "cpu", True, reason)
