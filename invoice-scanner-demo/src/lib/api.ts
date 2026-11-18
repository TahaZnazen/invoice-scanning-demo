import Constants from 'expo-constants';
import { File } from 'expo-file-system';

export type Pt = [number, number]; // normalised 0..1, corners ordered TL, TR, BR, BL
export type Mode = 'original' | 'color' | 'bw';

/** Mirrors ScanCreated in scan-service/app.py. */
export type ScanCreated = {
  id: string;
  photoWidth: number;
  photoHeight: number;
  corners: Pt[];
  detected: boolean;
  method: string;
  pageWidth: number;
  pageHeight: number;
  ms: number;
};

/**
 * Requests go to the Expo dev server the app was loaded from (host:8081), which forwards
 * /scan-api/* to the local scan service (see metro.config.js). One address, no extra port.
 * Override with EXPO_PUBLIC_SCAN_API=http://x.x.x.x:8000 to call the scan service directly.
 */
export const API_BASE = (() => {
  const override = process.env.EXPO_PUBLIC_SCAN_API;
  if (override) return override.replace(/\/$/, '');
  const devServer = Constants.expoConfig?.hostUri ?? Constants.expoGoConfig?.debuggerHost ?? 'localhost:8081';
  return `http://${devServer}/scan-api`;
})();

async function request<T>(path: string, init: RequestInit, timeoutMs: number): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { ...init, signal: ctrl.signal });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error ?? `HTTP ${res.status}`);
    return body as T;
  } catch (e: any) {
    if (e?.name === 'AbortError') throw new Error(`Timed out talking to ${API_BASE}`);
    if (e instanceof TypeError) throw new Error(`Can't reach the scan service at ${API_BASE}. Same Wi-Fi? Running?`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export const ping = () => request<{ ok: boolean }>('/health', { method: 'GET' }, 4000);

/** Uploads a photo (the file itself is the body: no base64, no multipart); the service finds the page. */
export const createScan = (photoUri: string) =>
  request<ScanCreated>(
    '/scans',
    {
      method: 'POST',
      headers: { 'Content-Type': 'image/jpeg' },
      body: new File(photoUri) as unknown as Blob, // expo-file-system File implements Blob
    },
    90000,
  );

const cornersParam = (corners: Pt[]) => corners.map(([x, y]) => `${x.toFixed(4)},${y.toFixed(4)}`).join(',');

/** URL of the cropped page. Same corners + mode = same URL, so <Image> caches it. */
export const pageUrl = (scanId: string, corners: Pt[], mode: Mode) =>
  `${API_BASE}/scans/${scanId}/page.jpg?mode=${mode}&corners=${cornersParam(corners)}`;
