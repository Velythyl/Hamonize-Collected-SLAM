from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_calibration(filepath: Path) -> dict:
    with filepath.open("r", encoding="utf-8") as handle:
        calib = yaml.safe_load(handle)

    if "camera_matrix" not in calib:
        raise ValueError(f"camera_matrix missing in {filepath}")

    camera_matrix = np.array(calib["camera_matrix"]["data"], dtype=np.float64).reshape(
        3, 3
    )
    distortion = np.array(
        calib.get("distortion_coefficients", {}).get("data", []), dtype=np.float64
    )

    width = calib.get("image_width")
    height = calib.get("image_height")
    if width is None or height is None:
        raise ValueError(f"image_width/height missing in {filepath}")

    return {
        "K": camera_matrix,
        "dist": distortion,
        "width": int(width),
        "height": int(height),
    }
