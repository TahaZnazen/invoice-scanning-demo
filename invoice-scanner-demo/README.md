# Invoice scanner demo

Photo of an invoice in, cropped flat page out. Nothing is saved.

```text
Scan invoice, Expo Go        -> iPhone camera photo -> corners suggested by scan-service :8000
                                -> you drag the 4 corners -> Crop -> cropped page
Scan invoice, dev build      -> Apple VisionKit document scanner (on the phone)            -> cropped pages
Choose from photos           -> scan-service                                               -> cropped page
```

Cropped pages from the scan service: Original (default) / Enhanced (colours kept) / B&W.

## Run

```bash
cd "/home/mehdi/Documents/OCR docs" && ./run-demo.sh   # --debug: save photos + outlines, -c: clear cache
```

Scan the QR code with the iPhone camera (opens in **Expo Go**, same Wi-Fi as the laptop).
In Expo Go, **Scan invoice** takes a normal photo and the scan service crops it.

With the development build installed (below), run `./run-demo.sh --dev-build` and open
**Invoice Scanner Demo** instead: **Scan invoice** then opens the iPhone's own document scanner.

## How a photo is cropped (scan service)

Edge-based candidates (outlines, straight lines, paper-coloured blobs) plus one from a trained
model, [DocCornerNet](https://github.com/marquaye/scanic) (MIT, ~2 MB ONNX, ~7 ms), all scored by
the same checks (edges along each side, paper brighter than outside, one paper colour inside).
The model's outline is also the answer when edges are too faint (white page on a white table).

Then a second opinion from [SAM 2.1](https://github.com/facebookresearch/sam2) (Meta's Segment
Anything, Apache-2.0, 126 MB ONNX, ~1.4 s on the laptop): it is asked which object is at the centre
of the photo, and replaces the outline when a side doesn't follow that object's edge (a fold, a
coloured block, a strip of desk). Downloaded by `run-demo.sh` on the first run.

## Code

```text
src/app/                 screens (Expo Router: one file per route)
  _layout.tsx            stack navigator + scan session provider
  index.tsx              home: Scan invoice (native scanner) / Choose from photos
  result.tsx             scanned pages, or the cropped photo (Original / Enhanced / B&W), Adjust
  crop.tsx               photo -> suggested corners -> drag them onto the invoice -> Crop
src/components/          PageImage, ResultHeader, CornerEditor, BusyOverlay
src/scan/ScanSession.tsx the scan on screen, shared by the screens
src/lib/
  documentScanner.ts     the phone's scanner (react-native-document-scanner-plugin: VisionKit / ML Kit)
  api.ts                 scan-service client (types mirror scan-service/app.py)
  photo.ts               shrink photo before upload, delete temp files
  pickPhoto.ts           photo library picker
```

## Later: the iPhone's own document scanner (development build, Apple Developer account $99/year)

The scanner is native code, so the app runs as a *development build* (your own "Expo Go" with
the scanner inside), built in the cloud by EAS. Needs an Apple Developer account ($99/year).

```bash
cd "/home/mehdi/Documents/OCR docs/invoice-scanner-demo"
npx eas-cli@latest init                    # link this project to your Expo account (once)
npx eas-cli@latest device:create           # register the iPhone: open the link it shows ON the iPhone
npx eas-cli@latest build --profile development --platform ios
#   sign in with your Apple Developer account when asked; EAS creates the certificates.
#   ~15-20 min; at the end, open the install link/QR on the iPhone.
```

On the iPhone: **Settings → Privacy & Security → Developer Mode → On** (restarts the phone).

Rebuild only when native packages change (`npx expo install <native lib>`); JavaScript changes
load live from the laptop like in Expo Go.

## Checks

```bash
npx tsc --noEmit && npx expo lint && npx expo-doctor
cd ../scan-service && .venv/bin/pytest -q
```
