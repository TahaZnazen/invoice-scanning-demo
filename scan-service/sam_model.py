"""Which object is at this point? SAM 2.1 (Meta's Segment Anything 2.1, hiera-tiny), ONNX.

Weights: Apache-2.0, exported to ONNX by samexporter, published at
https://huggingface.co/nrl-ai/samexporter-onnx-models (sam2_1_hiera_tiny/, ~126 MB, pinned
revision + checksums below). Runs with plain onnxruntime on the CPU: ~1.5 s per photo.

scanner.py asks for the object at the photo's centre and uses the confidence map to veto page
outlines that don't follow the object's boundary (a fold, the edge of a coloured block, a strip
of desk). Optional: detection works without it, as before.

    python sam_model.py --download      # fetch the model into models/sam2.1_hiera_tiny/
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from functools import cache
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "models" / "sam2.1_hiera_tiny"
REPO_URL = "https://huggingface.co/nrl-ai/samexporter-onnx-models/resolve/5050a79cd4b912dd745fff83047c4ef6fbd97be5"
ENCODER = "sam2.1_hiera_tiny.encoder.onnx"
DECODER = "sam2.1_hiera_tiny.decoder.onnx"
SHA256 = {
    ENCODER: "3d5ae796097a6afb1198ceea4187ecaa82736242d07f5658541829d6dfc3f871",
    DECODER: "197f1c7c8fd070aaf6f15ffc635f4c384422ba5eaa0f8c9e543e3d4b8e0e8e87",
}

INPUT_SIZE = 1024  # the encoder takes a 1024x1024 image (stretched, not padded)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ObjectModel:
    def __init__(self, folder: Path = MODEL_DIR) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(folder / ENCODER), opts, providers=providers)
        self.decoder = ort.InferenceSession(str(folder / DECODER), opts, providers=providers)

    def logits(self, rgb: np.ndarray, point: tuple[float, float], out_size: tuple[int, int]) -> np.ndarray:
        """Confidence map of the object at `point` (pixel coords in `rgb`), as (h, w) logits
        for out_size = (w, h): > 0 inside the object, higher = more confident. Of SAM's three
        guesses (part / bigger part / whole object) the largest is returned: the whole sheet,
        not one block printed on it."""
        h, w = rgb.shape[:2]
        # Shrink keeping the shape, then stretch to the square input. (Measured on the real
        # test photos: right corners on 16/19 this way, 14/19 stretching the photo directly.)
        s = INPUT_SIZE / max(h, w)
        x = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        x = cv2.resize(x, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        feats0, feats1, embedding = self.encoder.run(None, {"image": x})
        # One positive click, plus SAM's "no box" padding point (label -1).
        coords = np.array([[[point[0] * INPUT_SIZE / w, point[1] * INPUT_SIZE / h], [0.0, 0.0]]], np.float32)
        masks, _ = self.decoder.run(None, {
            "image_embed": embedding,
            "high_res_feats_0": feats0,
            "high_res_feats_1": feats1,
            "point_coords": coords,
            "point_labels": np.array([[1, -1]], np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), np.float32),
            "has_mask_input": np.zeros(1, np.float32),
        })
        masks = masks[0]  # (3, 256, 256) low-resolution logits
        whole = masks[int(np.argmax((masks > 0).sum(axis=(1, 2))))]
        return cv2.resize(whole, out_size, interpolation=cv2.INTER_LINEAR)


def is_downloaded(folder: Path = MODEL_DIR) -> bool:
    return all((folder / name).exists() for name in SHA256)


@cache
def get_sam() -> ObjectModel | None:
    """Shared instance; None (detection continues without it) if the model isn't downloaded."""
    return ObjectModel() if is_downloaded() else None


def download(folder: Path = MODEL_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, sha in SHA256.items():
        if (folder / name).exists():
            continue
        print(f"downloading {name}…")
        tmp = folder / (name + ".part")
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"{REPO_URL}/sam2_1_hiera_tiny/{name}") as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest() != sha:
            tmp.unlink()
            sys.exit(f"checksum mismatch for {name}: the file changed upstream")
        tmp.rename(folder / name)
    print(f"saved to {folder}")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    print("model ready" if is_downloaded() else "model missing: run with --download")
