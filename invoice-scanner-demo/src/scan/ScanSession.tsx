import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react';
import { createScan, type Mode, type Pt, type ScanCreated } from '@/lib/api';
import { deleteQuietly, shrinkForUpload } from '@/lib/photo';

/**
 * The scan on screen, shared by the screens. Two sources:
 *  - "scanned": pages from the phone's document scanner, already cropped and cleaned on the phone;
 *  - "photo":   a photo from the library, cropped by the scan service (corners adjustable).
 */
export type PhotoScan = {
  photoUri: string; // local photo (kept for the corner editor)
  status: 'uploading' | 'ready' | 'error';
  scan: ScanCreated | null;
  corners: Pt[];
  mode: Mode;
  error: string | null;
  elapsedMs: number | null;
};

export type Session = { kind: 'none' } | { kind: 'scanned'; pages: string[] } | ({ kind: 'photo' } & PhotoScan);

type Actions = {
  showScannedPages: (pages: string[]) => void;
  cropPhoto: (photoUri: string) => void;
  retry: () => void;
  setCorners: (corners: Pt[]) => void;
  setMode: (mode: Mode) => void;
};

const Context = createContext<{ session: Session } & Actions>(null as never);

export function ScanSessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>({ kind: 'none' });
  const current = useRef(0); // ignores the result of an upload the user already replaced

  const updatePhoto = useCallback(
    (token: number, patch: Partial<PhotoScan>) =>
      setSession((s) => (token === current.current && s.kind === 'photo' ? { ...s, ...patch } : s)),
    [],
  );

  const showScannedPages = useCallback((pages: string[]) => {
    current.current++;
    setSession({ kind: 'scanned', pages });
  }, []);

  const cropPhoto = useCallback(
    (photoUri: string) => {
      const token = ++current.current;
      setSession({
        kind: 'photo',
        photoUri,
        status: 'uploading',
        scan: null,
        corners: [],
        mode: 'original',
        error: null,
        elapsedMs: null,
      });
      const t0 = Date.now();
      (async () => {
        let upload = photoUri;
        try {
          upload = await shrinkForUpload(photoUri).catch(() => photoUri); // fall back to the original
          const scan = await createScan(upload);
          updatePhoto(token, { status: 'ready', scan, corners: scan.corners, elapsedMs: Date.now() - t0 });
        } catch (e: any) {
          updatePhoto(token, { status: 'error', error: e?.message ?? String(e) });
        } finally {
          if (upload !== photoUri) deleteQuietly(upload);
        }
      })();
    },
    [updatePhoto],
  );

  const retry = useCallback(() => {
    if (session.kind === 'photo') cropPhoto(session.photoUri);
  }, [session, cropPhoto]);

  const setCorners = useCallback(
    (corners: Pt[]) => setSession((s) => (s.kind === 'photo' ? { ...s, corners } : s)),
    [],
  );
  const setMode = useCallback((mode: Mode) => setSession((s) => (s.kind === 'photo' ? { ...s, mode } : s)), []);

  const value = useMemo(
    () => ({ session, showScannedPages, cropPhoto, retry, setCorners, setMode }),
    [session, showScannedPages, cropPhoto, retry, setCorners, setMode],
  );
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export const useScanSession = () => useContext(Context);
