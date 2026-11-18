import { useCallback, useEffect, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { API_BASE, ping } from '@/lib/api';
import { hasDocumentScanner, scanDocument } from '@/lib/documentScanner';
import { pickPhoto, takePhoto } from '@/lib/pickPhoto';
import { useScanSession } from '@/scan/ScanSession';
import { colors } from '@/theme';

type ServerStatus = 'checking' | 'ok' | 'down';

const STATUS_TEXT: Record<ServerStatus, string> = {
  checking: 'Checking scan service…',
  ok: 'Scan service connected',
  down: 'Scan service unreachable (tap to retry)',
};
const STATUS_COLOR: Record<ServerStatus, string> = { checking: colors.pending, ok: colors.ok, down: colors.error };

function useServerStatus() {
  const [status, setStatus] = useState<ServerStatus>('checking');
  const probe = useCallback(() => {
    ping().then(
      () => setStatus('ok'),
      () => setStatus('down'),
    );
  }, []);
  useEffect(probe, [probe]);
  const check = useCallback(() => {
    setStatus('checking');
    probe();
  }, [probe]);
  return { status, check };
}

export default function Home() {
  const { status, check } = useServerStatus();
  const { showScannedPages, cropPhoto } = useScanSession();

  /** The phone's document scanner if this build has it; otherwise a normal photo, cropped by the scan service. */
  async function scan() {
    if (!hasDocumentScanner) {
      const uri = await takePhoto();
      if (!uri) return;
      cropPhoto(uri);
      router.push('/crop');
      return;
    }
    try {
      const pages = await scanDocument();
      if (!pages) return;
      showScannedPages(pages);
      router.push('/result');
    } catch (e: any) {
      Alert.alert('Scanner error', e?.message ?? String(e));
    }
  }

  async function fromLibrary() {
    const uri = await pickPhoto();
    if (!uri) return;
    cropPhoto(uri);
    router.push('/crop');
  }

  return (
    <SafeAreaView style={styles.root}>
      <Text style={styles.title}>Invoice Scanner</Text>
      <Text style={styles.subtitle}>Demo · photo in, clean invoice out</Text>

      <Pressable style={styles.pill} onPress={check}>
        <View style={[styles.dot, { backgroundColor: STATUS_COLOR[status] }]} />
        <Text style={styles.pillText}>{STATUS_TEXT[status]}</Text>
      </Pressable>
      <Text style={styles.url}>{API_BASE}</Text>
      {!hasDocumentScanner && (
        <Text style={styles.notice}>
          Expo Go: photos are cropped by the scan service. The live document scanner needs the development build.
        </Text>
      )}

      <View style={{ flex: 1 }} />

      <Pressable style={styles.primary} onPress={scan}>
        <Text style={styles.primaryText}>Scan invoice</Text>
      </Pressable>
      <Pressable style={styles.secondary} onPress={fromLibrary}>
        <Text style={styles.secondaryText}>Choose from photos</Text>
      </Pressable>
      <Text style={styles.tips}>
        {hasDocumentScanner
          ? 'Hold the phone over the invoice: the scanner finds the page and captures it on its own.'
          : 'Lay the invoice flat, fit the whole page in the photo, avoid glare.'}
      </Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg, padding: 24 },
  title: { color: colors.text, fontSize: 32, fontWeight: '800', marginTop: 24 },
  subtitle: { color: colors.textMuted, fontSize: 16, marginTop: 4 },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 28,
    backgroundColor: colors.surface,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 999,
    alignSelf: 'flex-start',
  },
  dot: { width: 10, height: 10, borderRadius: 5 },
  pillText: { color: '#e5e7eb', fontSize: 14 },
  url: { color: colors.textFaint, fontSize: 12, marginTop: 6, marginLeft: 4 },
  notice: { color: colors.warn, fontSize: 13, marginTop: 16, lineHeight: 18 },
  primary: { backgroundColor: colors.primary, paddingVertical: 20, borderRadius: 16, alignItems: 'center' },
  primaryText: { color: colors.text, fontSize: 20, fontWeight: '700' },
  secondary: { paddingVertical: 16, borderRadius: 16, alignItems: 'center', marginTop: 12, backgroundColor: colors.surface },
  secondaryText: { color: '#e5e7eb', fontSize: 16, fontWeight: '600' },
  tips: { color: colors.textFaint, fontSize: 13, textAlign: 'center', marginTop: 20, lineHeight: 19 },
});
