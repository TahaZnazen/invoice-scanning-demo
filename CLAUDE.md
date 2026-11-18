# Invoice scanning demo: project state

Part of the product in `PRD-Accounting-E-Invoicing-Platform-v1.pdf` (Tunisian accounting firms +
their client businesses; e-invoicing via TTN). This repo is a **demo of PRD F4** (send supplier
invoices by photo, auto-crop): photo of an invoice in, cropped flat page out. **Nothing is saved** yet
(deliberately out of scope so far, along with login).

## Layout

```text
run-demo.sh               one command: scan service (bg, :8000, log -> scan-service/service.log) + Expo
                          --debug  save each photo + detected outline to scan-service/debug-photos/
                          --dev-build  target the dev build instead of Expo Go (default: --go)
invoice-scanner-demo/     Expo SDK 58 / RN 0.88 / Expo Router app (see its README.md and AGENTS.md)
scan-service/             Python 3.14 FastAPI + OpenCV + 2 ONNX models (DocCornerNet, SAM 2.1):
                          detects the page, renders the cropped page
```

## How it works today (user's setup: iPhone + this Linux laptop, Expo Go)

1. **Scan invoice** -> iPhone's normal camera (`expo-image-picker`), or **Choose from photos**.
2. Photo shrunk on the phone (3000 px, `expo-image-manipulator`) and uploaded as raw JPEG bytes to
   `POST /scans` (through the Expo dev server: `metro.config.js` proxies `/scan-api/*` -> :8000, so the
   phone needs only port 8081).
3. Service suggests 4 corners -> **crop screen** (`src/app/crop.tsx` + `CornerEditor`): user drags
   the corners (magnifier, dimmed outside) -> **Crop**.
4. Result: `GET /scans/{id}/page.jpg?mode=original|color|bw&corners=...` loaded by `<Image>`
   (cached, other modes prefetched). Default **Original**; "Enhanced" = gentle colour-preserving filter.

The **user confirmed the manual-corner flow works well** (2026-11-18).

### Native document scanner (ready, not usable yet)

`react-native-document-scanner-plugin` (Apple VisionKit / Google ML Kit) is installed and wired
(`src/lib/documentScanner.ts`): in a **development build**, Scan invoice opens the native scanner
instead (best quality). Needs EAS cloud build + **Apple Developer account ($99/yr): user can't pay
now**. Steps are in `invoice-scanner-demo/README.md` ("Later"). `eas.json` exists; `eas init` not run.

### Page detection (scan-service/scanner.py)

1. Candidates: closed edge outlines, pairs of Hough lines, paper-coloured blobs, plus one from a
   trained model **DocCornerNet** (`corner_model.py`, MIT, 1.9 MB ONNX, ~7 ms). Scored vectorised: edge
   support per side, paper brighter than outside, area, border-side penalty, then for the top 30:
   white-balanced Lab chroma checks (interior + rim). DocCornerNet's outline is the fallback. ~100 ms.
2. **SAM veto** (added 2026-11-19): **SAM 2.1 hiera-tiny** (`sam_model.py`, Meta, Apache-2.0, ONNX from
   HF `nrl-ai/samexporter-onnx-models`, pinned revision + sha256, 126 MB, ~1.4 s on this CPU) is asked
   for the object under one click at the photo centre. Its confidence map (logits) vetoes the chosen
   outline if a side has no confidence drop across it (fold, colour-block edge), lies in background
   (outline too big), or sits on the photo border with no page inside it. Vetoed -> the candidate (or
   SAM's own mask outline, sides refitted as lines) with the best IoU with SAM's mask, if IoU >= 0.9
   (hand/shadow over the page -> keep the old outline). method = "sam" when it replaced the outline.
   Optional: without the model files detection works as before.

- Result on the 19 labelled real photos (`tests/bench.py`): **16/19 within 3%, worst 7.4%** (before:
  10/19, worst 44%: all study-sheet fold / colour-block / cut-off cases). Same results across a
  threshold sweep. The 3 misses: paper on a light-blue folder in dim light (x2, SAM sees paper +
  folder as one object), thumb over the page edge.
- Fragile detail: SAM's input resize. Shrink keeping aspect, then stretch to 1024x1024 = 16/19; a
  direct stretch (SAM's official way) = 14/19.
- Tried and dropped: DocAligner (83 MB, outlines folders); SAM mask alone (always includes the
  folder: 6/19); MobileSAM / EfficientSAM-Ti instead of SAM 2.1 (10 and 13/19); SAM used to *rank*
  candidates (trims coloured headers, breaks synthetic cases); several clicks (worse); int8-quantised
  SAM encoder (1.0 s instead of 1.5 s, but 15/19 and threshold-sensitive).

## Next steps (agreed with the user)

1. **Better starting corners**: SAM veto done (above). **User tested it on the iPhone on several
   invoices (2026-11-19): "works good".** Next known failure, to fix later: **two stacked sheets**
   (one on top of another): the outline mixes both. Also left: paper on a near-same-colour folder in
   dim light, thumb over an edge (3 xfails). The user's test scans of 2026-11-19 are in
   `scan-service/debug-photos/00*` with their final corners: candidates for new tests. User wants
   **off-the-shelf libraries/models, not more hand-tuned rules**. Learning loop is in place: with
   `--debug` the service also saves the corners the user crops with (`<stem>-corners.json`);
   `tests/add_real_photo.py <debug file> <name>` turns one into a test case. Grow the test set
   that way before tuning more (19 photos is small; the numbers above are +-2 photos).
2. **B&W mode is too noisy** (`enhance_bw`: adaptive threshold + `flat < 225` term creates specks).
   Prefer an established binarisation (e.g. Sauvola/Wolf via a library) and/or despeckle.
3. Then PRD **F5**: AI reading of the cropped invoice (supplier, MF tax ID, number, date, HT, VAT by
   rate, stamp duty, TTC) + automatic checks. Then saving/duplicates (F4 rules).

## Commands

```bash
./run-demo.sh --debug                                   # then scan the QR with the iPhone camera
cd invoice-scanner-demo && npx tsc --noEmit && npx expo lint && npx expo-doctor
cd scan-service && .venv/bin/pytest -q                  # 44 pass + 3 xfail (known hard cases), ~80 s
cd scan-service && .venv/bin/python tests/bench.py      # error per real/synthetic photo (corner suggestion quality)
```

## Gotchas

- `expo-dev-client` is installed, so plain `npx expo start` targets the dev build: use `--go` for
  Expo Go (run-demo.sh does). QR "no usable data" on iPhone = wrong target or Expo Go missing.
- `react-dom` is pinned to the `react` version (19.3.0); otherwise `npx expo install` fails (ERESOLVE).
- Laptop IP changes (home Wi-Fi vs iPhone hotspot 172.20.10.x); Expo puts the current one in the QR.
  ufw is installed but disabled.
- System Node has no TypeScript stripping; system Python has no ensurepip (venv was made with pip
  available later). Installing unknown/unvetted packages may be blocked by the permission classifier."read CLAUDE.md and let's improve the starting corners".
- Real test photos in `scan-service/tests/real/` and `debug-photos/` contain personal data
  (gitignored: `debug-photos/`, `tests/real/*.jpg`). Expected corners: `tests/real/expected.json` (`_known_hard` = xfail with reason).
- Models live in `scan-service/models/` (gitignored); `run-demo.sh` downloads them on first run.
- Git: one repo at the project root, private on GitHub (`TahaZnazen/invoice-scanning-demo`).
  The Expo template's own `invoice-scanner-demo/.git` (1 template commit) was moved out on 2026-11-19.
