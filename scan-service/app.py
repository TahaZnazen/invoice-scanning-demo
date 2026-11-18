"""Scan service: photo in, flat cleaned-up invoice out.

    POST /scans                      body: JPEG/PNG/HEIC bytes  -> ScanCreated (corners, sizes)
    GET  /scans/{id}/page.jpg        ?mode=original|color|bw&corners=x,y,x,y,x,y,x,y -> JPEG
    GET  /health

Used for photos that didn't go through the phone's own document scanner
(library photos now; uploaded / emailed invoices later).

Run:  .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Set SCAN_DEBUG_DIR=<dir> to keep incoming photos + detected outline (for tuning detection),
and the corners the user finally cropped with (the right answer, for tests).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

import scanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("scan")

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
PHOTO_CACHE_SIZE = 8


# ---------- response models (the app's src/api.ts mirrors these) ----------


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ScanCreated(ApiModel):
    id: str
    photo_width: int
    photo_height: int
    corners: list[tuple[float, float]]
    detected: bool
    method: str
    page_width: int
    page_height: int
    ms: int


# ---------- state ----------


class PhotoStore:
    """Last few decoded photos, so corner/filter changes don't need a re-upload.
    In memory only: this demo deliberately saves nothing."""

    def __init__(self, size: int) -> None:
        self._photos: OrderedDict[str, np.ndarray] = OrderedDict()
        self._size = size

    def add(self, img: np.ndarray) -> str:
        scan_id = uuid.uuid4().hex
        self._photos[scan_id] = img
        while len(self._photos) > self._size:
            self._photos.popitem(last=False)
        return scan_id

    def get(self, scan_id: str) -> np.ndarray:
        img = self._photos.get(scan_id)
        if img is None:
            raise HTTPException(404, "photo expired, scan again")
        self._photos.move_to_end(scan_id)
        return img


class DebugRecorder:
    """Saves incoming photos + detected outline to a folder, for tuning detection. When a photo
    is then cropped, the corners the user settled on are saved next to it (-corners.json): the
    right answer for that photo, ready to become a test (tests/add_real_photo.py)."""

    def __init__(self, folder: str | None) -> None:
        self.folder = Path(folder) if folder else None
        self._scans: OrderedDict[str, tuple[Path, scanner.Detection, scanner.Corners | None]] = OrderedDict()

    def scan(self, scan_id: str, raw: bytes, img: np.ndarray, det: scanner.Detection) -> None:
        if not self.folder:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        stem = self.folder / f"{datetime.now():%H%M%S}-{scan_id[:8]}-{det.method}"
        stem.with_name(stem.name + "-photo.jpg").write_bytes(raw)
        outline = scanner.draw_outline(img, det.corners)
        stem.with_name(stem.name + "-outline.jpg").write_bytes(scanner.encode_jpeg(outline, 80))
        self._scans[scan_id] = (stem, det, None)
        while len(self._scans) > 64:
            self._scans.popitem(last=False)

    def cropped(self, scan_id: str, corners: scanner.Corners) -> None:
        """Called for every page render; the last corners asked for are the user's final choice."""
        if scan_id not in self._scans or self._scans[scan_id][2] == corners:
            return  # not recorded, or the same crop in another mode
        stem, det, _ = self._scans[scan_id]
        self._scans[scan_id] = (stem, det, corners)
        record = {"suggested": det.corners, "method": det.method, "final": corners}
        stem.with_name(stem.name + "-corners.json").write_text(json.dumps(record) + "\n")


photos = PhotoStore(PHOTO_CACHE_SIZE)
debug = DebugRecorder(os.environ.get("SCAN_DEBUG_DIR"))
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the models before the first request, not during it (SAM takes ~1 s to load).
    await run_in_threadpool(lambda: (scanner.get_model(), scanner.get_sam()))
    yield


app = FastAPI(title="Invoice scan service", lifespan=lifespan)


# ---------- helpers ----------


async def read_image(request: Request, max_side: int) -> tuple[bytes, np.ndarray]:
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "send the image bytes as the request body")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large")
    try:
        return raw, await run_in_threadpool(scanner.load_image, raw, max_side)
    except Exception as e:
        raise HTTPException(400, f"could not read image: {e}") from e


def parse_corners(text: str) -> scanner.Corners:
    try:
        v = [float(x) for x in text.split(",")]
    except ValueError:
        v = []
    if len(v) != 8 or not all(-0.01 <= x <= 1.01 for x in v):
        raise HTTPException(400, "corners must be 8 comma-separated numbers in 0..1 (TL, TR, BR, BL)")
    return [(min(1.0, max(0.0, v[i])), min(1.0, max(0.0, v[i + 1]))) for i in range(0, 8, 2)]


def elapsed_ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


# ---------- errors + logging: always {"error": "..."} ----------


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    return JSONResponse({"error": str(exc.errors()[0].get("msg", "invalid request"))}, status_code=422)


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse({"error": str(exc)}, status_code=500)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    if request.url.path != "/health" or response.status_code >= 400:
        log.info("%s %s -> %s (%dms)", request.method, request.url.path, response.status_code, elapsed_ms(t0))
    return response


# ---------- routes ----------


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/scans", response_model=ScanCreated)
async def create_scan(request: Request) -> ScanCreated:
    """Upload a photo: detects the page, keeps the photo in memory for page.jpg renders."""
    t0 = time.perf_counter()
    raw, img = await read_image(request, scanner.MAX_SIDE)
    det = await run_in_threadpool(scanner.detect_corners, img)
    scan_id = photos.add(img)
    h, w = img.shape[:2]
    page_w, page_h = scanner.page_size(w, h, det.corners)
    log.info("scan %s %dx%d %.1fMB method=%s", scan_id[:8], w, h, len(raw) / 1e6, det.method)
    await run_in_threadpool(debug.scan, scan_id, raw, img, det)
    return ScanCreated(
        id=scan_id,
        photo_width=w,
        photo_height=h,
        corners=det.corners,
        detected=det.detected,
        method=det.method,
        page_width=page_w,
        page_height=page_h,
        ms=elapsed_ms(t0),
    )


@app.get("/scans/{scan_id}/page.jpg")
async def page_image(scan_id: str, corners: str, mode: scanner.Mode = Query("original")) -> Response:
    """The flattened, cleaned page. Same URL = same image, so clients can cache it."""
    img = photos.get(scan_id)
    chosen = parse_corners(corners)
    debug.cropped(scan_id, chosen)
    out = await run_in_threadpool(scanner.render_scan, img, chosen, mode)
    jpeg = await run_in_threadpool(scanner.encode_jpeg, out)
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
