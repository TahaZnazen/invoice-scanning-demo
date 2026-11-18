"""HTTP routes: raw image bytes in, JSON / JPEG out, errors always {"error": ...}."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import scanner
from app import app
from photos import H, W, make_invoice, make_photo, max_corner_error, project_a4

JPEG = {"Content-Type": "image/jpeg"}
QUAD = project_a4(30, 6, 4, 520)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def photo_jpeg() -> bytes:
    return scanner.encode_jpeg(make_photo(make_invoice(), QUAD, [70, 55, 45], 0.3))


def corners_param(corners) -> str:
    return ",".join(f"{v:.4f}" for p in corners for v in p)


def test_scan_then_render_every_mode(client, photo_jpeg):
    r = client.post("/scans", content=photo_jpeg, headers=JPEG)
    assert r.status_code == 200, r.text
    scan = r.json()
    assert scan["detected"] and scan["method"] != "none"
    assert max_corner_error(scan["corners"], [(x / W, y / H) for x, y in QUAD]) < 0.02
    for mode in scanner.MODES:
        page = client.get(f"/scans/{scan['id']}/page.jpg", params={"mode": mode, "corners": corners_param(scan["corners"])})
        assert page.status_code == 200 and page.headers["content-type"] == "image/jpeg"
        size = Image.open(io.BytesIO(page.content)).size
        assert abs(size[0] - scan["pageWidth"]) <= 1 and abs(size[1] - scan["pageHeight"]) <= 1


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status"),
    [
        ("post", "/scans", {"content": b"not an image", "headers": JPEG}, 400),
        ("post", "/scans", {"content": b""}, 400),
        ("post", "/scans", {"content": b"\xff\xd8 broken", "headers": JPEG}, 400),
        ("get", "/scans/unknown/page.jpg", {"params": {"corners": "0,0,1,0,1,1,0,1"}}, 404),
    ],
)
def test_errors_are_json(client, method, url, kwargs, status):
    r = getattr(client, method)(url, **kwargs)
    assert r.status_code == status and r.json()["error"]


def test_bad_corners_and_mode(client, photo_jpeg):
    scan_id = client.post("/scans", content=photo_jpeg, headers=JPEG).json()["id"]
    url = f"/scans/{scan_id}/page.jpg"
    assert client.get(url, params={"corners": "1,2,3"}).status_code == 400
    assert client.get(url, params={"corners": "0,0,1,0,1,1,0,1", "mode": "sepia"}).status_code == 422


def test_page_defaults_to_original(client, photo_jpeg):
    scan = client.post("/scans", content=photo_jpeg, headers=JPEG).json()
    url = f"/scans/{scan['id']}/page.jpg"
    c = corners_param(scan["corners"])
    plain = client.get(url, params={"corners": c})
    assert plain.content == client.get(url, params={"corners": c, "mode": "original"}).content


def test_debug_mode_records_the_final_crop(tmp_path, monkeypatch, photo_jpeg):
    """--debug: photo + suggestion saved on upload, the user's final corners on crop, and
    tests/add_real_photo.py turns that into a test case."""
    import json

    import add_real_photo
    from app import DebugRecorder

    rec = DebugRecorder(str(tmp_path / "debug"))
    det = scanner.Detection([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], True, "lines")
    rec.scan("abcdef0123456789", photo_jpeg, scanner.load_image(photo_jpeg), det)
    final = [(0.12, 0.1), (0.9, 0.11), (0.88, 0.9), (0.1, 0.9)]
    rec.cropped("abcdef0123456789", final)
    rec.cropped("unknown-scan", final)  # photos uploaded before a restart: ignored
    (record_file,) = (tmp_path / "debug").glob("*-abcdef01-lines-corners.json")
    record = json.loads(record_file.read_text())
    assert record["final"] == [list(p) for p in final] and record["suggested"] == [list(p) for p in det.corners]

    monkeypatch.setattr(add_real_photo, "REAL", tmp_path / "real")
    add_real_photo.add(str(record_file), "my-invoice")
    assert (tmp_path / "real" / "my-invoice.jpg").read_bytes() == photo_jpeg
    assert json.loads((tmp_path / "real" / "expected.json").read_text())["my-invoice.jpg"][0] == [0.12, 0.1]
