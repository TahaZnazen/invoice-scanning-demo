"""CamScanner-style pipeline: detect page -> perspective warp -> enhance.

All functions work on RGB uint8 numpy arrays. Corners are normalised (0..1),
ordered top-left, top-right, bottom-right, bottom-left.
"""

from __future__ import annotations

import io
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

from corner_model import get_model
from sam_model import get_sam

try:  # iPhone photos picked from the library can be HEIC
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

MAX_SIDE = 3000  # working resolution of the photo (long edge)
DETECT_SIDE = 900  # detection runs on a small copy: faster and less noisy
MAX_OUT_SIDE = 2600  # cap on the scanned page size
MIN_PAGE_AREA = 0.15  # page must cover at least 15% of the photo
PHONE_FOCAL = 0.75  # focal length / long side for a ~26mm-equivalent phone camera

Mode = Literal["original", "color", "bw"]
MODES: tuple[Mode, ...] = ("original", "color", "bw")
Corners = list[tuple[float, float]]
Quads = np.ndarray  # (N, 4, 2) float64, each quad ordered TL, TR, BR, BL

FALLBACK_CORNERS: Corners = [(0.03, 0.03), (0.97, 0.03), (0.97, 0.97), (0.03, 0.97)]


@dataclass(frozen=True)
class Detection:
    corners: Corners
    detected: bool
    method: str  # "edges" | "lines" | "paper-colour" | "model" | "sam" | "none"


# ---------- image IO ----------


_EXIF_ROTATION = {3: cv2.ROTATE_180, 6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}


def load_image(data: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """Decode JPEG/PNG/HEIC, apply EXIF rotation, downscale to max_side, return RGB."""
    img = Image.open(io.BytesIO(data))
    scale = min(1.0, max_side / max(img.size))
    if img.format == "JPEG" and scale < 1:
        # JPEG can decode directly at 1/2, 1/4 or 1/8 size: pick the smallest still >= target.
        img.draft("RGB", (math.ceil(img.width * scale), math.ceil(img.height * scale)))
    orientation = img.getexif().get(0x0112, 1)
    if orientation not in (1, *_EXIF_ROTATION):  # mirrored variants: rare, let Pillow do them
        img, orientation = ImageOps.exif_transpose(img), 1
    arr = np.asarray(img.convert("RGB"))
    if orientation in _EXIF_ROTATION:
        arr = cv2.rotate(arr, _EXIF_ROTATION[orientation])
    h, w = arr.shape[:2]
    if (s := max_side / max(h, w)) < 1:
        arr = cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    bgr = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# ---------- quad geometry (batched over N quads) ----------


def order_quads(q: Quads) -> Quads:
    """Order each quad's corners TL, TR, BR, BL (clockwise in image coords)."""
    c = q.mean(axis=1, keepdims=True)
    angle = np.arctan2(q[..., 1] - c[..., 1], q[..., 0] - c[..., 0])
    q = np.take_along_axis(q, np.argsort(angle, axis=1)[..., None], axis=1)
    start = np.argmin(q.sum(axis=2), axis=1)
    return np.take_along_axis(q, ((start[:, None] + np.arange(4)) % 4)[..., None], axis=1)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Single-quad version of order_quads."""
    return order_quads(np.asarray(pts, dtype=np.float64).reshape(1, 4, 2))[0]


def quad_areas(q: Quads) -> np.ndarray:
    x, y = q[..., 0], q[..., 1]
    return 0.5 * np.abs((x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1)).sum(axis=1))


def plausible_quads(q: Quads) -> np.ndarray:
    """Every interior angle between 45 and 135 degrees: rejects slivers and odd shapes."""
    v1 = np.roll(q, 1, axis=1) - q
    v2 = np.roll(q, -1, axis=1) - q
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return ((deg >= 45) & (deg <= 135)).all(axis=1)


def _side_points(q: Quads, t: np.ndarray) -> np.ndarray:
    """Points along each side at fractions t: (N, 4, len(t), 2)."""
    a = q[:, :, None, :]
    b = np.roll(q, -1, axis=1)[:, :, None, :]
    return a + (b - a) * t[None, None, :, None]


def _across_sides(q: Quads, values: np.ndarray, t: np.ndarray, offset: float):
    """`values` (an image) sampled `offset` px inside and outside each side, at fractions t
    along it: (N, 4, len(t)) each, plus which outside samples fall within the image."""
    h, w = values.shape[:2]
    pts = _side_points(q, t)
    d = np.roll(q, -1, axis=1) - q
    d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
    normal = np.stack([d[..., 1], -d[..., 0]], axis=-1)[:, :, None, :]  # outward (clockwise corners, y down)
    inside = (pts - offset * normal).round().astype(np.int32)
    outside = (pts + offset * normal).round().astype(np.int32)  # round before the bounds check
    valid = (outside[..., 0] >= 0) & (outside[..., 0] < w) & (outside[..., 1] >= 0) & (outside[..., 1] < h)
    v_in = values[inside[..., 1].clip(0, h - 1), inside[..., 0].clip(0, w - 1)].astype(np.float32)
    v_out = values[outside[..., 1].clip(0, h - 1), outside[..., 0].clip(0, w - 1)].astype(np.float32)
    return v_in, v_out, valid


# ---------- detection ----------
#
# 1. Candidates: many possible page outlines, from three sources:
#      "edges"        closed outlines in the edge map,
#      "lines"        pairs of long straight horizontal + vertical edges (robust when the page
#                     touches other objects, which breaks outline tracing),
#      "paper-colour" bright, unsaturated blobs.
# 2. Scoring, vectorised over all candidates: how much of each side lies on a real edge,
#    whether the paper is brighter than just outside each side, size, a penalty for sides
#    along the photo border; then, for the best few, whether the inside is one paper colour.

MIN_SIDE_SUPPORT = 0.45  # weakest side must sit on edges for >= 45% of its length
MAX_AREA = 0.985  # bigger = the photo frame itself
BORDER_SIDE_PENALTY = 0.85  # score factor per side that only runs along the photo border
UNIFORMITY_POWER = 3  # how strongly a two-coloured interior (page + folder) is punished
UNIFORMITY_TOP_N = 30  # the (slowest) colour check only runs on the best few
UNIFORMITY_SCALE = 1 / 3
UNIFORMITY_TOLERANCE = 7  # Lab a/b units
WHITE_PATCH_PERCENTILE = 97  # brightest 3% of the frame = the paper's white, for white balance
RIM_WIDTH = 0.05  # band just inside the sides checked for paper colour, as a fraction of the quad
RIM_POWER = 2
SUPPORT_SAMPLES = 80
CONTRAST_SAMPLES = 40
CONTRAST_OFFSET = 8  # px (at detection scale) inside / outside each side
MAX_LINES = 8  # strongest straight lines kept per direction


def _prep(small: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # Closing wipes out printed text and table lines so mostly the page outline produces edges.
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)
    med = float(np.median(blurred))
    edge_maps = [
        cv2.Canny(blurred, lo, hi) for lo, hi in ((0.66 * med, min(255.0, 1.33 * med)), (20, 60), (50, 150))
    ]
    return blurred, edge_maps


def _contour_to_quad(cnt: np.ndarray, rect_fallback: bool) -> np.ndarray | None:
    """Reduce a contour to 4 corners via its convex hull."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64) if rect_fallback else None


def _contour_quads(mask: np.ndarray, rect_fallback: bool, mode: int = cv2.RETR_LIST) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_PAGE_AREA * mask.shape[0] * mask.shape[1] * 0.5
    quads = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        if cv2.contourArea(cnt) < min_area:
            break
        if (q := _contour_to_quad(cnt, rect_fallback)) is not None:
            quads.append(q)
    return quads


def _edge_candidates(edge_maps: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for edges in edge_maps:
        e = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        cv2.rectangle(e, (0, 0), (e.shape[1] - 1, e.shape[0] - 1), 255, 1)  # closes pages touching the border
        out += _contour_quads(e, rect_fallback=False)
    return out


def _paper_candidates(small: np.ndarray) -> list[np.ndarray]:
    """Bright + unsaturated blobs. Opening with growing kernels cuts the page loose from
    other white things touching it (keyboard, wall, second sheet)."""
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    white = cv2.GaussianBlur(cv2.subtract(hsv[:, :, 2], hsv[:, :, 1]), (7, 7), 0)
    _, mask = cv2.threshold(white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    out = []
    for k in (9, 25, 45, 75):
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))  # rect: separable, fast
        out += _contour_quads(opened, rect_fallback=True, mode=cv2.RETR_EXTERNAL)
    return out


def _cluster_lines(segs: np.ndarray) -> np.ndarray:
    """Merge Hough segments lying on the same straight line. Returns the strongest lines
    as rows (nx, ny, c) with nx*x + ny*y = c and (nx, ny) a unit normal."""
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    cos_6deg = math.cos(math.radians(6))
    clusters: list[dict] = []
    for i in np.argsort(-lengths):
        x1, y1, x2, y2 = segs[i].astype(np.float64)
        n = np.array([-(y2 - y1), x2 - x1]) / (lengths[i] + 1e-9)
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for c in clusters:
            if abs(np.dot(c["n"], n)) > cos_6deg and abs(np.dot(c["n"], mid) - c["c"]) < 12:
                c["len"] += lengths[i]
                c["pts"] += [(x1, y1), (x2, y2)]
                break
        else:
            clusters.append({"n": n, "c": float(np.dot(n, mid)), "len": lengths[i], "pts": [(x1, y1), (x2, y2)]})
    lines = []
    for c in sorted(clusters, key=lambda c: -c["len"])[:MAX_LINES]:
        vx, vy, x0, y0 = cv2.fitLine(np.array(c["pts"], np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((-vy, vx, -vy * x0 + vx * y0))
    return np.array(lines, dtype=np.float64).reshape(-1, 3)


def _line_candidates(edges: np.ndarray) -> np.ndarray:
    """Quads from every pair of near-horizontal x pair of near-vertical straight lines."""
    h, w = edges.shape
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=0.12 * min(h, w), maxLineGap=25)
    if segs is None:
        return np.empty((0, 4, 2))
    segs = segs.reshape(-1, 4)
    horizontal = np.abs(segs[:, 2] - segs[:, 0]) >= np.abs(segs[:, 3] - segs[:, 1])
    # Photo borders are fallback sides, for pages cut off by the frame.
    hs = np.vstack([_cluster_lines(segs[horizontal]), [(0, 1, 0), (0, 1, h - 1)]])
    vs = np.vstack([_cluster_lines(segs[~horizontal]), [(1, 0, 0), (1, 0, w - 1)]])

    # All horizontal x vertical intersections at once (Cramer's rule): (H, V, 2).
    a1, b1, c1 = (hs[:, None, k] for k in range(3))
    a2, b2, c2 = (vs[None, :, k] for k in range(3))
    det = a1 * b2 - a2 * b1
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.stack([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det], axis=-1)
    m = 0.05 * max(h, w)
    ok = (np.abs(det) > 1e-6) & (p[..., 0] >= -m) & (p[..., 0] <= w + m) & (p[..., 1] >= -m) & (p[..., 1] <= h + m)

    # Every (top, bottom) x (left, right) combination.
    hi, hj = np.triu_indices(len(hs), k=1)
    vi, vj = np.triu_indices(len(vs), k=1)
    hh, vv = (g.ravel() for g in np.meshgrid(np.arange(len(hi)), np.arange(len(vi)), indexing="ij"))
    t, b, l, r = hi[hh], hj[hh], vi[vv], vj[vv]
    valid = ok[t, l] & ok[t, r] & ok[b, r] & ok[b, l]
    q = np.stack([p[t, l], p[t, r], p[b, r], p[b, l]], axis=1)[valid]
    q = q[quad_areas(q) >= MIN_PAGE_AREA * h * w]
    q[..., 0] = q[..., 0].clip(0, w - 1)
    q[..., 1] = q[..., 1].clip(0, h - 1)
    return order_quads(q)


def _side_support(q: Quads, edge_support: np.ndarray) -> np.ndarray:
    """(N, 4): fraction of each side lying on (near) an edge pixel. Sides running along the
    photo border get a neutral 0.5: the page may be cut off there, but it's no evidence."""
    h, w = edge_support.shape
    pts = _side_points(q, np.linspace(0.05, 0.95, SUPPORT_SAMPLES)).round().astype(np.int32)
    xs, ys = pts[..., 0].clip(0, w - 1), pts[..., 1].clip(0, h - 1)
    on_border = (xs <= 3) | (xs >= w - 4) | (ys <= 3) | (ys >= h - 4)
    return np.where(on_border, 0.5, edge_support[ys, xs] > 0).mean(axis=-1)


def _side_contrast(q: Quads, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 4) paper-just-inside vs just-outside brightness per side (0..1, 0.5 = equal), and
    (N,) number of sides on the photo border (can't be checked: neutral 0.5)."""
    lum_in, lum_out, valid = _across_sides(q, gray, np.linspace(0.1, 0.9, CONTRAST_SAMPLES), CONTRAST_OFFSET)
    border = valid.mean(axis=-1) < 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN sides are border sides anyway
        med = np.nanmedian(np.where(valid, lum_in - lum_out, np.nan), axis=-1)
    contrast = np.where(border, 0.5, np.clip(0.5 + np.nan_to_num(med) / 80, 0, 1))
    return contrast, border.sum(axis=1)


def _white_balance(rgb: np.ndarray) -> np.ndarray:
    """'White patch' correction: scale each channel so the brightest pixels (the paper) become
    neutral. Under a coloured light (bluish room, warm bulb) white paper otherwise takes the
    light's colour and can't be told apart from a light-blue folder or a beige desk."""
    lum = rgb.max(axis=2)
    bright = rgb[lum >= np.percentile(lum, WHITE_PATCH_PERCENTILE)]
    ref = bright.mean(axis=0) if len(bright) else np.full(3, 255.0)
    gain = ref.max() / np.maximum(ref, 1.0)
    return np.clip(rgb * gain, 0, 255).astype(np.uint8)


def _paper_consistency(q: np.ndarray, ab: np.ndarray) -> float:
    """Is the quad one sheet of paper? Product of two checks against the interior's median
    chroma (Lab a/b only, so shadows, which darken but don't change colour, don't count):
      - interior: fraction of the inside with that colour (rejects page + keyboard/table),
      - rim: fraction of a thin band just inside the sides with that colour. A page lying on
        a folder or mat gets outlined with a strip of folder along some sides; that strip is
        a small part of the interior but most of the rim on those sides."""
    pts = (q * UNIFORMITY_SCALE).round().astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)  # work inside the quad's bounding box only
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + bw, ab.shape[1]), min(y + bh, ab.shape[0])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, pts - (x0, y0), 1)
    k = max(3, round(RIM_WIDTH * min(bw, bh)) | 1)
    core = cv2.erode(mask, np.ones((k, k), np.uint8))
    inside, rim = mask.view(bool), (mask & ~core).view(bool)
    a, b = ab[y0:y1, x0:x1, 0], ab[y0:y1, x0:x1, 1]
    if np.count_nonzero(inside) < 50 or np.count_nonzero(rim) < 20:
        return 0.0
    ma, mb = np.median(a[inside]), np.median(b[inside])
    same = (np.abs(a - ma) <= UNIFORMITY_TOLERANCE) & (np.abs(b - mb) <= UNIFORMITY_TOLERANCE)
    interior = np.count_nonzero(same & inside) / np.count_nonzero(inside)
    rim_ok = np.count_nonzero(same & rim) / np.count_nonzero(rim)
    return interior**UNIFORMITY_POWER * rim_ok**RIM_POWER


def _outline_candidates(q: Quads, methods: np.ndarray, edge_support: np.ndarray):
    """The plausible outlines: big enough, page-like angles, every side mostly on an edge.
    Returns (quads, methods, area as a fraction of the photo, (N, 4) edge support per side)."""
    h, w = edge_support.shape
    area = quad_areas(q) / (h * w)
    keep = (area >= MIN_PAGE_AREA) & plausible_quads(q)
    q, methods, area = q[keep], methods[keep], area[keep]
    support = _side_support(q, edge_support)
    keep = support.min(axis=1) >= MIN_SIDE_SUPPORT
    return q[keep], methods[keep], area[keep], support[keep]


def _pick_best(
    q: Quads, methods: np.ndarray, area: np.ndarray, support: np.ndarray, small: np.ndarray, blurred: np.ndarray
):
    """Returns (quad, method) of the best candidate, or None."""
    keep = area <= MAX_AREA
    q, methods, area, support = q[keep], methods[keep], area[keep], support[keep]
    if not len(q):
        return None

    contrast, border_sides = _side_contrast(q, blurred)
    score = (
        (0.5 * support.min(axis=1) + 0.5 * support.mean(axis=1))
        * (0.25 + 0.5 * contrast.min(axis=1) + 0.5 * contrast.mean(axis=1))
        # area weighs fully: dark bands *inside* the invoice (table headers) also look like
        # page edges, and the quad they form is always smaller than the real page.
        * area
        * BORDER_SIDE_PENALTY**border_sides
    )

    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    tiny = _white_balance(cv2.resize(closed, None, fx=UNIFORMITY_SCALE, fy=UNIFORMITY_SCALE, interpolation=cv2.INTER_AREA))
    ab = cv2.cvtColor(tiny, cv2.COLOR_RGB2LAB)[:, :, 1:].astype(np.int16)
    top = np.argsort(-score, kind="stable")[:UNIFORMITY_TOP_N]
    final = [score[k] * _paper_consistency(q[k], ab) for k in top]
    k = top[int(np.argmax(final))]
    return q[k], str(methods[k])


# ---------- second opinion: what object is in the middle of the photo? ----------
#
# The checks above judge outlines by local evidence (edges, brightness, colour), so a straight
# edge *inside* the page can fool them: a fold, the edge of a coloured block, or a strip of desk
# next to the page. SAM 2.1 (sam_model.py, a trained segmentation model) is asked for the object
# at the photo's centre. Its confidence map vetoes the chosen outline if one of its sides is
#   - not an object boundary: SAM is about as confident on both sides (fold, colour block),
#   - in the background: SAM is not confident just inside it (outline too big),
#   - on the photo border with no page just inside it (claims a cut-off page that isn't there).
# A vetoed outline is replaced by the candidate (or SAM's own outline) that best matches SAM's
# mask, provided that mask is page-shaped: a hand or a shadow across the page makes it not, and
# then the outline from the checks is kept.
# Set on the 19 real test photos (tests/bench.py); same results for EDGE_DROP 0.7-1.3,
# SAMPLE_OFFSET 1.5-3 %, MIN_FIT 0.85-0.93 and BACKGROUND down to -2 (one photo worse at -3.5).

SAM_SAMPLE_OFFSET = 0.02  # how far inside / outside each side to look, as a fraction of the photo
SAM_SAMPLES = 30
SAM_EDGE_DROP = 1.0  # logits: the drop across a side needed for it to be an object boundary
SAM_BACKGROUND = -2.5  # logits: a side whose inside is below this lies in the background
SAM_MIN_FIT = 0.9  # IoU of SAM's mask with the replacement outline


def _sam_rejects(quad: np.ndarray, logits: np.ndarray) -> bool:
    """True if SAM's confidence map contradicts any side of the outline (see above)."""
    h, w = logits.shape
    t = np.linspace(0.15, 0.85, SAM_SAMPLES)
    v_in, v_out, valid = _across_sides(quad[None], logits, t, SAM_SAMPLE_OFFSET * max(h, w))
    for inside, outside, ok in zip(v_in[0], v_out[0], valid[0]):
        if ok.mean() < 0.5:  # side along the photo border: the page must continue past it
            if np.median(inside) < 0:
                return True
            continue
        lin, lout = np.median(inside), np.median(outside[ok])
        if lin < SAM_BACKGROUND or (lin > 0 and lin - lout < SAM_EDGE_DROP):
            return True
    return False


def _fit_sides(quad: np.ndarray, pts: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Refit each side of `quad` as a straight line through the outline points along its middle
    80%, and intersect neighbouring sides: corners follow the straight parts of a mask's outline,
    not its rounded tips."""
    a, b = quad, np.roll(quad, -1, axis=0)
    d = b - a
    t = ((pts[:, None] - a[None]) * d[None]).sum(-1) / (d**2).sum(-1)[None]  # (P, 4)
    proj = a[None] + t.clip(0, 1)[..., None] * d[None]
    side = np.linalg.norm(pts[:, None] - proj, axis=-1).argmin(axis=1)
    t_own = t[np.arange(len(pts)), side]
    lines = []
    for k in range(4):
        p = pts[(side == k) & (t_own > 0.1) & (t_own < 0.9)].astype(np.float32)
        lines.append(cv2.fitLine(p, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel() if len(p) >= 10 else None)
    h, w = shape[:2]
    out = quad.copy()
    for k in range(4):  # corner k joins side k-1 and side k
        l1, l2 = lines[k - 1], lines[k]
        if l1 is None or l2 is None:
            continue
        A = np.array([[l1[0], -l2[0]], [l1[1], -l2[1]]])
        if abs(np.linalg.det(A)) < 1e-6:
            continue
        s = np.linalg.solve(A, l2[2:] - l1[2:])
        p = l1[2:] + s[0] * l1[:2]
        if np.linalg.norm(p - quad[k]) < 0.1 * max(h, w):
            out[k] = p
    out[:, 0] = out[:, 0].clip(0, w - 1)
    out[:, 1] = out[:, 1].clip(0, h - 1)
    return out


def _mask_quad(mask: np.ndarray) -> np.ndarray | None:
    """4-corner outline of the biggest blob in a mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    quad = order_corners(_contour_to_quad(cnt, rect_fallback=True))
    return _fit_sides(quad, cnt.reshape(-1, 2).astype(np.float64), mask.shape)


def _mask_iou(q: Quads, mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    buf = np.zeros(m.shape, np.uint8)
    out = np.empty(len(q))
    for i, quad in enumerate(q):
        buf[:] = 0
        cv2.fillConvexPoly(buf, quad.round().astype(np.int32), 1)
        inside = buf.view(bool)
        inter = np.count_nonzero(inside & m)
        out[i] = inter / max(1, np.count_nonzero(inside | m))
    return out


def _sam_review(img: np.ndarray, quad: np.ndarray | None, candidates: Quads, size: tuple[int, int]):
    """SAM's second opinion on `quad` (detection-scale px, None if nothing was found).
    Returns a replacement outline, or None to keep `quad`."""
    sam = get_sam()
    if sam is None:
        return None
    h, w = img.shape[:2]
    logits = sam.logits(img, (w / 2, h / 2), size)
    if quad is not None and not _sam_rejects(quad, logits):
        return None
    mask = logits > 0
    if (mq := _mask_quad(mask)) is not None:
        candidates = np.concatenate([candidates, mq[None]])
    if quad is None:  # nothing found by the checks: on a blank photo SAM's "object" is the whole frame
        candidates = candidates[quad_areas(candidates) / (size[0] * size[1]) <= MAX_AREA]
    if not len(candidates):
        return None
    fit = _mask_iou(candidates, mask)
    k = int(np.argmax(fit))
    return candidates[k] if fit[k] >= SAM_MIN_FIT else None


def detect_corners(img: np.ndarray) -> Detection:
    """Finds the page in a photo (for photos that didn't go through the phone's scanner)."""
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_SIDE / max(w, h))
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    blurred, edge_maps = _prep(small)
    model = get_model()
    model_quad = model(small) if model else None  # trained model: one more candidate, and the fallback
    groups = {
        "edges": _edge_candidates(edge_maps),
        "lines": _line_candidates(cv2.dilate(edge_maps[0] | edge_maps[1], None)),
        "paper-colour": _paper_candidates(small),
        "model": [np.array(model_quad) * (sw, sh)] if model_quad else [],
    }
    arrays = [order_quads(np.asarray(g, dtype=np.float64).reshape(-1, 4, 2)) for g in groups.values()]
    edge_support = cv2.dilate(np.bitwise_or.reduce(edge_maps), np.ones((7, 7), np.uint8))
    q, methods, area, support = _outline_candidates(
        np.concatenate(arrays), np.repeat(list(groups), [len(a) for a in arrays]), edge_support
    )

    if (best := _pick_best(q, methods, area, support, small, blurred)) is not None:
        quad, method = best
    elif model_quad:  # edges too faint for the checks, but the model sees a document
        quad, method = np.array(model_quad) * (sw, sh), "model"
    else:
        quad, method = None, "none"
    if (replacement := _sam_review(img, quad, q, (sw, sh))) is not None:
        quad, method = replacement, "sam"

    if quad is None:
        return Detection(FALLBACK_CORNERS, False, "none")
    corners = [(float(np.clip(x / sw, 0, 1)), float(np.clip(y / sh, 0, 1))) for x, y in quad]
    return Detection(corners, True, method)


# ---------- warp + enhance ----------


def true_aspect_ratio(q: np.ndarray, img_w: int, img_h: int) -> float | None:
    """Real width/height of the page, undoing perspective foreshortening
    (Zhang & He, "Whiteboard scanning and image enhancement", 2007).
    Uses a typical phone focal length instead of estimating it, which is unstable
    when the photo is taken nearly straight on."""
    tl, tr, br, bl = (np.array([x, y, 1.0]) for x, y in q)
    u0, v0, f = img_w / 2, img_h / 2, PHONE_FOCAL * max(img_w, img_h)
    k2 = np.dot(np.cross(tl, br), bl) / np.dot(np.cross(tr, br), bl)
    k3 = np.dot(np.cross(tl, br), tr) / np.dot(np.cross(bl, br), tr)
    n2 = k2 * tr - tl
    n3 = k3 * bl - tl

    def unproject(n: np.ndarray) -> np.ndarray:
        return np.array([(n[0] - u0 * n[2]) / f, (n[1] - v0 * n[2]) / f, n[2]])

    ar = float(np.linalg.norm(unproject(n2)) / np.linalg.norm(unproject(n3)))
    return ar if math.isfinite(ar) and 0.2 < ar < 5 else None


def _pixel_corners(img_w: int, img_h: int, corners: Corners) -> np.ndarray:
    return np.array([(x * (img_w - 1), y * (img_h - 1)) for x, y in corners], dtype=np.float64)


def page_size(img_w: int, img_h: int, corners: Corners) -> tuple[int, int]:
    """Output (width, height) of the flattened page, with its real proportions."""
    q = _pixel_corners(img_w, img_h, corners)
    tl, tr, br, bl = q
    out_w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    out_h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if ar := true_aspect_ratio(q, img_w, img_h):
        if ar < out_w / out_h:
            out_h = out_w / ar
        else:
            out_w = out_h * ar
    s = min(1.0, MAX_OUT_SIDE / max(out_w, out_h))
    return max(16, round(out_w * s)), max(16, round(out_h * s))


def warp(img: np.ndarray, corners: Corners) -> np.ndarray:
    h, w = img.shape[:2]
    out_w, out_h = page_size(w, h, corners)
    dst = np.array([(0, 0), (out_w - 1, 0), (out_w - 1, out_h - 1), (0, out_h - 1)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(_pixel_corners(w, h, corners).astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def flatten_lighting(img: np.ndarray) -> np.ndarray:
    """Estimate the paper background (lighting + shadows) and divide it out, so the
    paper becomes uniformly white while ink keeps its colour."""
    h, w = img.shape[:2]
    sw, sh = max(8, round(w * 0.25)), max(8, round(h * 0.25))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    # Max-filter radius ~3% of the page: swallows solid blocks (table headers, logos)
    # while still following shadows.
    k = max(5, round(max(sw, sh) * 0.03) | 1)
    bg = cv2.medianBlur(cv2.dilate(small, np.ones((k, k), np.uint8)), 21)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(img, bg, scale=255)


SHADOW_MAX_GAIN = 1.6  # "Enhanced" brightens shadows on the paper at most this much
SHADOW_KERNEL = 0.06  # background estimate radius, as a fraction of the page


def enhance_colour(warped: np.ndarray) -> np.ndarray:
    """Gentle clean-up that keeps colours (logos, stamps, coloured paper):
    white balance from the paper, lift shadows using brightness only (same gain on every
    channel, capped, so hues don't shift), stretch contrast a little, light sharpening."""
    img = _white_balance(warped).astype(np.float32)
    lum = img.max(axis=2)
    h, w = lum.shape
    small = cv2.resize(lum, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(5, round(max(small.shape) * SHADOW_KERNEL) | 1)
    bg = cv2.GaussianBlur(cv2.dilate(small, np.ones((k, k), np.uint8)), (0, 0), k / 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    paper = np.percentile(bg, 90)
    gain = np.clip(paper / np.maximum(bg, 1), 1.0, SHADOW_MAX_GAIN)[..., None]
    img = img * gain
    black = min(np.percentile(img.max(axis=2), 1), 60.0)  # ink level, never lifted above 60
    img = (img - black) * (255.0 / max(paper - black, 1))
    out = np.clip(img, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
    return cv2.addWeighted(out, 1.3, blurred, -0.3, 0)


def enhance_bw(warped: np.ndarray) -> np.ndarray:
    flat = flatten_lighting(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY))
    block = max(15, round(flat.shape[1] / 60) | 1)
    adaptive = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 12)
    # Ink = clearly dark, OR locally darker than its surroundings (faint thermal print).
    ink = (flat < 150) | ((adaptive == 0) & (flat < 225))
    return np.where(ink, 0, 255).astype(np.uint8)


def render_scan(img: np.ndarray, corners: Corners, mode: Mode) -> np.ndarray:
    warped = warp(img, corners)
    if mode == "original":
        return warped
    return enhance_bw(warped) if mode == "bw" else enhance_colour(warped)


def draw_outline(img: np.ndarray, corners: Corners, width: int = 900) -> np.ndarray:
    """Debug helper: photo with the detected outline drawn on it."""
    h, w = img.shape[:2]
    pts = np.array([(x * w, y * h) for x, y in corners], dtype=np.int32)
    out = cv2.polylines(img.copy(), [pts], True, (255, 0, 64), thickness=max(4, w // 250))
    return cv2.resize(out, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
