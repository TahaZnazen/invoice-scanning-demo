"""Turn a real scan into a test: copies the photo from debug-photos/ to tests/real/<name>.jpg and
records the corners you cropped it with on the phone (saved by ./run-demo.sh --debug as
<stem>-corners.json) in tests/real/expected.json.

    .venv/bin/python tests/add_real_photo.py debug-photos/101502-1a2b3c4d-sam-photo.jpg <name>

Check the crop was right first (the result screen showed the whole invoice, nothing else).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REAL = Path(__file__).parent / "real"
SUFFIXES = ("-photo.jpg", "-outline.jpg", "-corners.json")


def add(debug_file: str, name: str) -> None:
    path = Path(debug_file)
    stem = path.name
    for suffix in SUFFIXES:
        stem = stem.removesuffix(suffix)
    corners_file = path.with_name(stem + "-corners.json")
    if not corners_file.exists():
        sys.exit(f"{corners_file} not found: crop this photo on the phone first (with --debug)")
    final = json.loads(corners_file.read_text())["final"]
    name = name.removesuffix(".jpg") + ".jpg"
    REAL.mkdir(exist_ok=True)
    shutil.copyfile(path.with_name(stem + "-photo.jpg"), REAL / name)
    expected_path = REAL / "expected.json"
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    expected[name] = [[round(x, 3), round(y, 3)] for x, y in final]
    if "_known_hard" in expected:  # keep the notes last
        expected["_known_hard"] = expected.pop("_known_hard")
    expected_path.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"added tests/real/{name}: {expected[name]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    add(sys.argv[1], sys.argv[2])
