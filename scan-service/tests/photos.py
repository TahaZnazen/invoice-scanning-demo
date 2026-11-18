"""Test photos: synthetic "phone photos" of an invoice, and the real ones in tests/real/."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import scanner

W, H = 3024, 4032  # synthetic photo size (12 MP portrait, like an iPhone)
A4 = 297 / 210

REAL_DIR = Path(__file__).parent / "real"  # real photos: personal data, not committed
_expected = json.loads((REAL_DIR / "expected.json").read_text()) if (REAL_DIR / "expected.json").exists() else {}
REAL_CORNERS: dict[str, list] = {k: v for k, v in _expected.items() if not k.startswith("_")}
REAL_KNOWN_HARD: dict[str, str] = _expected.get("_known_hard", {})  # name -> why it fails today


def make_invoice() -> np.ndarray:
    img = Image.new("RGB", (1000, 1414), (251, 250, 246))
    d = ImageDraw.Draw(img)

    def font(size: int):
        return ImageFont.load_default(size=size)

    d.text((80, 70), "FACTURE", fill=(0, 0, 0), font=font(54))
    d.text((80, 150), "N° FA-2026-00042 · Date: 17/11/2026", fill=(60, 60, 60), font=font(22))
    d.text((600, 100), "SARL Exemple\nMF: 1234567/A/M/000\nTunis, Tunisie", fill=(40, 40, 40), font=font(22))
    d.rectangle((80, 400, 920, 440), fill=(43, 95, 168))
    d.text((90, 408), "Désignation                                  Qté          Montant", fill="white", font=font(22))
    for i in range(12):
        y = 452 + i * 42
        d.text((80, y), f"Article {i + 1} - Prestation de service", fill=(0, 0, 0), font=font(22))
        d.text((620, y), str((i + 1) * 3), fill=(0, 0, 0), font=font(22))
        d.text((760, y), f"{123.45 * (i + 1):.3f} TND", fill=(0, 0, 0), font=font(22))
    d.line((80, 1000, 920, 1000), fill=(50, 50, 50), width=2)
    d.text((560, 1030), "Total HT: 9 629.100 TND\nTVA 19%: 1 829.529 TND\nTimbre: 1.000 TND", fill=(0, 0, 0), font=font(24))
    d.text((560, 1160), "TTC: 11 459.629 TND", fill=(0, 0, 0), font=font(28))
    d.ellipse((160, 1060, 340, 1240), outline=(32, 64, 192), width=6)
    return np.asarray(img)


def project_a4(tilt: float, yaw: float, roll: float, dist: float) -> list[tuple[int, int]]:
    """Corners of a 210x297mm sheet rotated in 3D, seen through a phone-like pinhole camera."""
    f = scanner.PHONE_FOCAL * H
    t, yw, r = map(math.radians, (tilt, yaw, roll))
    out = []
    for x, y, z in ((-105, -148.5, 0), (105, -148.5, 0), (105, 148.5, 0), (-105, 148.5, 0)):
        y, z = y * math.cos(t) - z * math.sin(t), y * math.sin(t) + z * math.cos(t)
        x, z = x * math.cos(yw) + z * math.sin(yw), -x * math.sin(yw) + z * math.cos(yw)
        x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
        z += dist
        out.append((round(W / 2 + f * x / z), round(H / 2 + f * y / z)))
    return out


def make_photo(page: np.ndarray, quad, bg_rgb, shadow: float, seed: int = 0) -> np.ndarray:
    """The page warped onto a noisy "table" with a hand/phone shadow."""
    rng = np.random.default_rng(seed)
    photo = np.clip(np.full((H, W, 3), bg_rgb, np.float32) + rng.normal(0, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    ph, pw = page.shape[:2]
    M = cv2.getPerspectiveTransform(np.float32([(0, 0), (pw - 1, 0), (pw - 1, ph - 1), (0, ph - 1)]), np.float32(quad))
    warped = cv2.warpPerspective(page, M, (W, H))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (W, H)) > 0
    photo[mask] = warped[mask]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    k = 1 - shadow * np.maximum(0, xx / W * 0.6 + yy / H * 0.6 - 0.3)
    photo = (photo * k[..., None]).astype(np.uint8)
    return cv2.GaussianBlur(photo, (3, 3), 0)


# name: (background RGB, page corners in px, shadow strength, is a camera-modelled A4 sheet)
CASES = {
    # freehand quads: detection only
    "dark-wood-tilted": ([92, 64, 40], [(520, 640), (2520, 820), (2380, 3560), (380, 3330)], 0.35, False),
    "grey-desk-straight": ([120, 124, 128], [(300, 500), (2700, 520), (2720, 3700), (280, 3680)], 0.2, False),
    "light-table-hard": ([205, 200, 190], [(700, 900), (2400, 760), (2600, 3300), (600, 3500)], 0.45, False),
    "strong-perspective": ([60, 70, 85], [(900, 700), (2200, 720), (2900, 3800), (150, 3750)], 0.3, False),
    # camera-modelled A4 sheets: the output ratio must be A4
    "camera-tilt-30": ([70, 55, 45], project_a4(30, 6, 4, 520), 0.3, True),
    "camera-tilt-45": ([110, 110, 115], project_a4(45, -8, -5, 480), 0.25, True),
    "camera-overhead": ([40, 60, 50], project_a4(5, 3, 12, 470), 0.2, True),
}


def make_folded_colourful_invoice(fold_y: float = 0.42) -> np.ndarray:
    """The invoice with big coloured blocks and a horizontal fold (the half below it lit a bit
    differently, the crease darker): straight edges *inside* the page, like a folded study sheet."""
    img = Image.fromarray(make_invoice())
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle((60, 40, w - 60, 130), fill=(60, 90, 170))  # header band
    d.rectangle((60, 560, w - 60, 640), fill=(185, 80, 60))  # red band
    d.rectangle((60, 700, w // 2 - 10, 860), fill=(80, 60, 140))  # purple block
    d.rectangle((w // 2 + 10, 700, w - 60, 860), fill=(30, 120, 90))  # green block
    a = np.asarray(img).astype(np.float32)
    y = int(fold_y * h)
    a[y:] *= 0.93
    a[y - 2 : y + 2] *= 0.6
    return np.clip(a, 0, 255).astype(np.uint8)


def visible_corners(quad) -> list[tuple[float, float]]:
    """Normalised corners of a page whose bottom may run off the photo: the bottom corners
    become the points where its sides leave the frame (where a user would put them)."""

    def clip(top, bottom):
        (x1, y1), (x2, y2) = top, bottom
        if y2 <= H - 1:
            return x2 / W, y2 / H
        return (x1 + (H - 1 - y1) / (y2 - y1) * (x2 - x1)) / W, (H - 1) / H

    tl, tr, br, bl = quad
    return [(tl[0] / W, tl[1] / H), (tr[0] / W, tr[1] / H), clip(tr, br), clip(tl, bl)]


# Folded page with colour blocks, running off the bottom of the photo: the fold and the block
# edges make a convincing smaller "page" for the edge checks (the study-sheet failure).
FOLDED_CUTOFF = ([45, 50, 60], [(500, 700), (2550, 640), (2850, 4300), (300, 4350)], 0.3)


def synthetic_cases() -> dict[str, tuple[np.ndarray, list]]:
    """name -> (photo, true normalised corners), for tests/bench.py."""
    invoice = make_invoice()
    out = {name: (make_photo(invoice, quad, bg, shadow), [(x / W, y / H) for x, y in quad])
           for name, (bg, quad, shadow, _) in CASES.items()}
    white_quad = project_a4(25, 5, 3, 520)
    for table in ([236, 234, 230], [248, 247, 245]):
        out[f"white-table-{table[0]}"] = (make_photo(invoice, white_quad, table, 0.0), [(x / W, y / H) for x, y in white_quad])
    bg, quad, shadow = FOLDED_CUTOFF
    out["folded-colour-blocks-cutoff"] = (make_photo(make_folded_colourful_invoice(), quad, bg, shadow), visible_corners(quad))
    return out


def max_corner_error(a, b) -> float:
    """Largest distance between matching corners (normalised coords)."""
    return max(math.hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(a, b))
