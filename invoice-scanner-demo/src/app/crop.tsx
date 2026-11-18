import { Image, Pressable, StyleSheet, Text } from 'react-native';
import { Redirect, router } from 'expo-router';
import { SafeAreaView } from 'react-native-safe-area-context';
import { BusyOverlay } from '@/components/BusyOverlay';
import { CornerEditor } from '@/components/CornerEditor';
import { useScanSession } from '@/scan/ScanSession';
import { colors } from '@/theme';

/**
 * Step 2 for a photo: the scan service suggests the 4 corners, the user drags them onto the
 * invoice's real corners, then Crop. (Result -> "Adjust" comes back here.)
 */
export default function CropScreen() {
  const { session, retry, setCorners } = useScanSession();
  if (session.kind !== 'photo') return <Redirect href="/" />;

  if (session.status === 'error') {
    return (
      <SafeAreaView style={[styles.root, styles.center]}>
        <Text style={styles.errorTitle}>Couldn&apos;t process the photo</Text>
        <Text style={styles.errorText}>{session.error}</Text>
        <Pressable style={styles.primary} onPress={retry}>
          <Text style={styles.primaryText}>Try again</Text>
        </Pressable>
        <Pressable onPress={() => router.back()} hitSlop={12} style={{ marginTop: 18 }}>
          <Text style={styles.link}>Back</Text>
        </Pressable>
      </SafeAreaView>
    );
  }

  if (session.status === 'uploading' || !session.scan) {
    return (
      <SafeAreaView style={styles.root}>
        <BusyOverlay label="Finding the invoice…">
          <Image source={{ uri: session.photoUri }} style={styles.busyPhoto} resizeMode="contain" />
        </BusyOverlay>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <CornerEditor
        key={session.scan.id} // new photo -> start again from its suggestion
        uri={session.photoUri}
        photoWidth={session.scan.photoWidth}
        photoHeight={session.scan.photoHeight}
        initial={session.corners}
        onBack={() => router.back()}
        onCrop={(corners) => {
          setCorners(corners);
          router.push('/result');
        }}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { alignItems: 'center', justifyContent: 'center', padding: 32 },
  busyPhoto: { width: '70%', height: '45%', opacity: 0.6, marginBottom: 12 },
  errorTitle: { color: colors.text, fontSize: 20, fontWeight: '700' },
  errorText: { color: colors.textMuted, fontSize: 14, marginTop: 8, textAlign: 'center' },
  primary: { marginTop: 20, backgroundColor: colors.primary, paddingHorizontal: 22, paddingVertical: 12, borderRadius: 12 },
  primaryText: { color: colors.text, fontWeight: '700' },
  link: { color: colors.link, fontSize: 16, fontWeight: '600' },
});
