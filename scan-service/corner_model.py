"""Page corners from a trained model: DocCornerNet LEAN (MobileNetV2 + SimCC head, 456K params).

Ships in the `scanic-ml` npm package (MIT): https://github.com/marquaye/scanic. Model card:
https://cdn.jsdelivr.net/npm/scanic-ml@0.2.0/MODEL_CARD.md. Run with plain onnxruntime (CPU, ~7 ms).

Used by scanner.py as one more candidate outline (the usual checks still decide), and as the
answer when the edge-based detector finds nothing. Always returns a quad, even for faint edges.

    python corner_model.py --download      # fetch the model (~1.9 MB) into models/
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import urllib.request
from functools import cache
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "doccornernet_lean.ort"
PACKAGE_URL = "https://registry.npmjs.org/scanic-ml/-/scanic-ml-0.2.0.tgz"
PACKAGE_MEMBER = "package/dist/doccornernet_lean.ort"
MODEL_SHA256 = "732eb8d02d83cc63efb3f9467f01e0c458205a89b2547de3870bc6db1a80c8b3"

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_DOC_SCORE = 0.5  # P(document present)


class CornerModel:
    def __init__(self, path: Path = MODEL_PATH) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def __call__(self, rgb: np.ndarray) -> list[tuple[float, float]] | None:
        """4 corners (normalised 0..1, TL, TR, BR, BL), or None if no document is seen."""
        x = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD)[None]  # NHWC
        coords, score_logit = self.session.run(["coords", "score_logit"], {"input": x})
        if 1 / (1 + np.exp(-float(score_logit.ravel()[0]))) < MIN_DOC_SCORE:
            return None
        return [(float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1))) for cx, cy in coords.reshape(4, 2)]


@cache
def get_model() -> CornerModel | None:
    """Shared instance; None (detection continues without it) if the model file is missing."""
    return CornerModel() if MODEL_PATH.exists() else None


def download(path: Path = MODEL_PATH) -> None:
    print("downloading DocCornerNet from the scanic-ml npm package…")
    with urllib.request.urlopen(PACKAGE_URL) as r, tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz") as tar:
        data = tar.extractfile(PACKAGE_MEMBER).read()  # read the one file we need; nothing is unpacked
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        sys.exit("checksum mismatch: the package changed upstream")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"saved {path}")


if __name__ == "__main__":
    if "--download" in sys.argv and not MODEL_PATH.exists():
        download()
    print("model ready" if MODEL_PATH.exists() else "model missing: run with --download")
