from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class DetectorPosition:
    lid: int
    x_km: float
    y_km: float
    z_km: float
    heicm: int = 0


def default_const_dst_path() -> Path | None:
    configured = os.environ.get("TALESD_CONST_DST")
    if configured:
        return Path(configured).expanduser()
    tadir = os.environ.get("TADIR")
    if tadir:
        candidate = Path(tadir).expanduser() / "data" / "SD" / "talesdconst_pass2.dst"
        if candidate.exists():
            return candidate
    return None


def parse_lid(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        value = value.strip().replace("SD", "").replace("sd", "")
    return int(value)


def load_tale_const_positions(path: str | Path | None = None) -> dict[int, DetectorPosition]:
    """Load TALE-SD detector positions from a talesdconst calibration DST."""

    if path is None:
        path = default_const_dst_path()
    if path is None:
        raise ValueError("TALE-SD const DST is required for MC input; pass --const-dst or set TALESD_CONST_DST/TADIR")

    const_path = Path(path).expanduser()
    if not const_path.exists():
        raise FileNotFoundError(f"TALE-SD const DST not found: {const_path}")

    import dstio

    detectors: dict[int, DetectorPosition] = {}
    with dstio.open(str(const_path), banks=["talesdconst"]) as dst:
        for event in dst:
            bank = event.get("talesdconst")
            if not bank:
                continue
            lid = parse_lid(bank["lid"])
            x = float(bank["posX"]) / 1.0e3
            y = float(bank["posY"]) / 1.0e3
            z = float(bank["posZ"]) / 1.0e3
            heicm = int(bank.get("heicm", 0))
            if lid <= 0:
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if abs(x) > 100.0 or abs(y) > 100.0 or abs(z) > 10.0:
                continue
            if heicm < 0:
                continue
            detectors[lid] = DetectorPosition(lid=lid, x_km=x, y_km=y, z_km=z, heicm=heicm)
    if not detectors:
        raise ValueError(f"no detector positions found in {const_path}")
    return detectors
