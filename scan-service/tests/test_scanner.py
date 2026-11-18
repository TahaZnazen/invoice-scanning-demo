"""Detection + rendering on synthetic photos and the real phone photos in tests/real/.

Run: .venv/bin/pytest -q          (set SYNTH_OUT=<dir> to keep the generated images)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import scanner
from photos import A4, CASES, FOLDED_CUTOFF, H, REAL_CORNERS, REAL_DIR, REAL_KNOWN_HARD, W
from photos import make_folded_colourful_invoice, make_invoice, make_photo, max_corner_error, project_a4, visible_corners

OUT = Path(os.environ["SYNTH_OUT"]) if os.environ.get("SYNTH_OUT") else None


@pytest.fixture(scope="module")
def invoice() -> np.ndarray:
    return make_invoice()


@pytest.mark.parametrize("name", CASES)
def test_synthetic_photo(invoice, name):
    bg, quad, shadow, is_a4 = CASES[name]
    photo = make_photo(invoice, quad, bg, shadow)
    det = scanner.detect_corners(photo)
    out = scanner.render_scan(photo, det.corners, "color")
    if OUT:
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / f"{name}-photo.jpg"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / f"{name}-color.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / f"{name}-bw.jpg"), scanner.render_scan(photo, det.corners, "bw"))
    assert det.detected, f"page not found (method={det.method})"
    truth = [(x / W, y / H) for x, y in quad]
    assert max_corner_error(det.corners, truth) * H < 40, "corners off by more than 40px"
    if is_a4:
        ratio = out.shape[0] / out.shape[1]
        assert abs(ratio - A4) < 0.06, f"h/w={ratio:.2f}, expected {A4:.2f}"


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.xfail(reason=REAL_KNOWN_HARD[n], strict=False)) if n in REAL_KNOWN_HARD else n
     for n in sorted(REAL_CORNERS)],
)
def test_real_photo(name):
    path = REAL_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    det = scanner.detect_corners(scanner.load_image(path.read_bytes()))
    assert det.detected
    assert max_corner_error(det.corners, REAL_CORNERS[name]) < 0.03, det.corners


def test_order_corners_any_input_order():
    pts = np.array([(10, 90), (90, 10), (10, 10), (90, 90)])
    assert scanner.order_corners(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_page_size_matches_render(invoice):
    photo = make_photo(invoice, CASES["camera-tilt-30"][1], [70, 55, 45], 0.3)
    det = scanner.detect_corners(photo)
    out = scanner.render_scan(photo, det.corners, "bw")
    assert scanner.page_size(photo.shape[1], photo.shape[0], det.corners) == (out.shape[1], out.shape[0])


def test_blank_photo_falls_back():
    det = scanner.detect_corners(np.full((800, 600, 3), 128, np.uint8))
    assert not det.detected and det.method == "none" and det.corners == scanner.FALLBACK_CORNERS


@pytest.mark.parametrize("table", [[236, 234, 230], [248, 247, 245]])
def test_white_page_on_white_table_uses_model(invoice, table):
    """Edges too faint for the edge-based detector: the trained model still finds the page."""
    if scanner.get_model() is None:
        pytest.skip("model not downloaded (python corner_model.py --download)")
    quad = project_a4(25, 5, 3, 520)
    det = scanner.detect_corners(make_photo(invoice, quad, table, 0.0))
    assert det.detected and det.method == "model"
    assert max_corner_error(det.corners, [(x / W, y / H) for x, y in quad]) < 0.03


# ---------- the SAM veto (scanner._sam_rejects / _sam_review) ----------


def _rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64)


def _logits(*layers) -> np.ndarray:
    """A hand-made SAM confidence map (400 x 300): -4 background, then (rect, value) layers."""
    m = np.full((400, 300), -4.0, np.float32)
    for (x0, y0, x1, y1), v in layers:
        m[y0:y1, x0:x1] = v
    return m


@pytest.mark.parametrize(
    ("layers", "outline", "rejected"),
    [
        # page on a plain desk: its own outline is fine
        ([((50, 50, 250, 350), 3.0)], _rect(50, 50, 250, 350), False),
        # top half of the page (a fold): the bottom side has the page on both sides
        ([((50, 50, 250, 350), 3.0)], _rect(50, 50, 250, 200), True),
        # outline too big: its sides run through the desk
        ([((50, 50, 250, 350), 3.0)], _rect(20, 20, 280, 380), True),
        # bottom side on the photo border, but the page ends above it
        ([((50, 50, 250, 350), 3.0)], _rect(50, 50, 250, 399), True),
        # page really running off the bottom of the photo
        ([((50, 50, 250, 400), 3.0)], _rect(50, 50, 250, 399), False),
        # paper on a folder: SAM is less sure about the folder strip, so the paper's sides are edges
        ([((30, 30, 270, 370), 1.0), ((50, 50, 250, 350), 3.0)], _rect(50, 50, 250, 350), False),
    ],
    ids=["page", "fold", "too-big", "fake-cutoff", "real-cutoff", "paper-on-folder"],
)
def test_sam_veto_rules(layers, outline, rejected):
    assert scanner._sam_rejects(outline, _logits(*layers)) is rejected


def test_mask_quad_follows_straight_sides():
    mask = np.zeros((400, 300), np.uint8)
    corners = np.array([(60, 40), (240, 60), (250, 350), (40, 330)], np.int32)
    cv2.fillConvexPoly(mask, corners, 1)
    assert np.abs(scanner._mask_quad(mask) - corners).max() < 3


def test_folded_page_running_off_photo():
    """The study-sheet failure: a fold + colour blocks make a smaller convincing "page" for the
    edge checks; SAM sees one sheet and vetoes it."""
    if scanner.get_sam() is None:
        pytest.skip("SAM not downloaded (python sam_model.py --download)")
    bg, quad, shadow = FOLDED_CUTOFF
    det = scanner.detect_corners(make_photo(make_folded_colourful_invoice(), quad, bg, shadow))
    assert det.detected
    assert max_corner_error(det.corners, visible_corners(quad)) < 0.03, (det.method, det.corners)
