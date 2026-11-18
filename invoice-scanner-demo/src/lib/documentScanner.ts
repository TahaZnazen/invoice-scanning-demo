import { TurboModuleRegistry } from 'react-native';

/**
 * The phone's own document scanner: Apple VisionKit on iOS, Google ML Kit on Android.
 * It finds the page, auto-captures, crops, flattens and lets the user pick a filter.
 *
 * It's native code, so it exists only in the development build, not in Expo Go.
 */
export const hasDocumentScanner = TurboModuleRegistry.get('DocumentScanner') != null;

/** Opens the scanner. Returns the scanned pages (file:// uris), or null if cancelled. */
export async function scanDocument(): Promise<string[] | null> {
  // Loaded lazily: importing it in Expo Go would throw at startup (module not found).
  const { default: scanner } = await import('react-native-document-scanner-plugin');
  const { status, scannedImages } = await scanner.scanDocument({ croppedImageQuality: 90 });
  return status === 'success' && scannedImages?.length ? scannedImages : null;
}
