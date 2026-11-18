import { useEffect } from 'react';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, View, useWindowDimensions } from 'react-native';
import { Redirect, router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { PageImage } from '@/components/PageImage';
import { ResultHeader } from '@/components/ResultHeader';
import { pageUrl, type Mode } from '@/lib/api';
import { scanDocument } from '@/lib/documentScanner';
import { useScanSession, type PhotoScan } from '@/scan/ScanSession';
import { colors } from '@/theme';

export default function ResultScreen() {
  const { session } = useScanSession();
  if (session.kind === 'none') return <Redirect href="/" />;
  return (
    <SafeAreaView style={styles.root}>
      {session.kind === 'scanned' ? <ScannedPages pages={session.pages} /> : <CroppedPhoto photo={session} />}
    </SafeAreaView>
  );
}

/** Pages from the phone's document scanner: already cropped, flattened and filtered. */
function ScannedPages({ pages }: { pages: string[] }) {
  const { showScannedPages } = useScanSession();
  const { width } = useWindowDimensions();

  async function rescan() {
    try {
      const next = await scanDocument();
      if (next) showScannedPages(next);
    } catch (e: any) {
      Alert.alert('Scanner error', e?.message ?? String(e));
    }
  }

  return (
    <>
      <ResultHeader
        title={pages.length === 1 ? 'Scanned invoice' : `${pages.length} pages`}
        left={{ label: 'Done', onPress: () => router.back() }}
        right={{ label: 'Rescan', onPress: rescan }}
      />
      <ScrollView contentContainerStyle={styles.pages} maximumZoomScale={5} minimumZoomScale={1}>
        {pages.map((uri) => (
          <PageImage key={uri} uri={uri} width={width - 24} />
        ))}
      </ScrollView>
      <Text style={styles.hint}>Pinch to zoom · filter chosen in the scanner (Color, Grayscale, B&W, Photo)</Text>
    </>
  );
}

const MODES: { key: Mode; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'color', label: 'Enhanced' },
  { key: 'bw', label: 'B&W' },
];

/** A photo cropped by the scan service with the corners the user set. Filters rendered server-side. */
function CroppedPhoto({ photo }: { photo: PhotoScan }) {
  const { setMode } = useScanSession();
  const { width } = useWindowDimensions();
  const { scan, corners, mode } = photo;

  // Fetch the other filters in the background: switching is instant.
  useEffect(() => {
    if (!scan) return;
    for (const m of MODES) if (m.key !== mode) Image.prefetch(pageUrl(scan.id, corners, m.key)).catch(() => {});
  }, [scan, corners, mode]);

  if (!scan) return <Redirect href="/crop" />;

  return (
    <>
      <ResultHeader
        title="Cropped invoice"
        left={{ label: 'Adjust', onPress: () => router.back() }}
        right={{ label: 'Done', onPress: () => router.dismissAll() }}
      />
      <ScrollView contentContainerStyle={styles.pages} maximumZoomScale={5} minimumZoomScale={1}>
        <PageImage uri={pageUrl(scan.id, corners, mode)} width={width - 24} initialAspect={scan.pageHeight / scan.pageWidth} />
      </ScrollView>
      <View style={styles.segment}>
        {MODES.map((m) => (
          <Pressable key={m.key} style={[styles.segBtn, mode === m.key && styles.segBtnActive]} onPress={() => setMode(m.key)}>
            <Text style={[styles.segText, mode === m.key && styles.segTextActive]}>{m.label}</Text>
          </Pressable>
        ))}
      </View>
      <Text style={styles.hint}>Pinch to zoom and check the text is sharp</Text>
    </>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  pages: { padding: 12, gap: 12 },
  segment: {
    flexDirection: 'row',
    marginHorizontal: 16,
    marginTop: 8,
    backgroundColor: colors.surface,
    borderRadius: 12,
    padding: 4,
  },
  segBtn: { flex: 1, paddingVertical: 10, borderRadius: 9, alignItems: 'center' },
  segBtnActive: { backgroundColor: colors.primary },
  segText: { color: colors.textMuted, fontWeight: '600' },
  segTextActive: { color: colors.text },
  hint: { color: colors.textFaint, fontSize: 12, textAlign: 'center', paddingVertical: 8, paddingHorizontal: 16 },
});
