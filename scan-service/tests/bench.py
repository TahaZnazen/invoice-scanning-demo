"""How close are the suggested corners? Runs detect_corners on every labelled real photo
(tests/real/expected.json) and on the synthetic photos, and prints each one's error: the
distance of the worst corner, as a fraction of the photo (< 0.03 counts as right).

    .venv/bin/python tests/bench.py            # all photos
    .venv/bin/python tests/bench.py study      # only names containing "study"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]

import numpy as np  # noqa: E402

import scanner  # noqa: E402
from photos import REAL_CORNERS, REAL_DIR, REAL_KNOWN_HARD, max_corner_error, synthetic_cases  # noqa: E402

GOOD = 0.03


def real_cases() -> dict[str, tuple[np.ndarray, list]]:
    return {
        name: (scanner.load_image((REAL_DIR / name).read_bytes()), truth)
        for name, truth in sorted(REAL_CORNERS.items())
        if (REAL_DIR / name).exists()
    }


def run(cases: dict[str, tuple[np.ndarray, list]], detect=scanner.detect_corners) -> dict[str, float]:
    errors = {}
    for name, (img, truth) in cases.items():
        t0 = time.perf_counter()
        det = detect(img)
        ms = (time.perf_counter() - t0) * 1000
        errors[name] = e = max_corner_error(det.corners, truth)
        flag = "  " if e < GOOD else "✗ "
        hard = "  (known hard)" if name in REAL_KNOWN_HARD else ""
        print(f"  {flag}{name:42s} {e:.3f}  {det.method:12s} {ms:5.0f} ms{hard}")
    good = sum(e < GOOD for e in errors.values())
    print(f"  -> {good}/{len(errors)} right, mean error {np.mean(list(errors.values())):.3f}, worst {max(errors.values()):.3f}")
    return errors


if __name__ == "__main__":
    pick = sys.argv[1] if len(sys.argv) > 1 else ""
    print("real photos")
    run({k: v for k, v in real_cases().items() if pick in k})
    print("synthetic photos")
    run({k: v for k, v in synthetic_cases().items() if pick in k})
