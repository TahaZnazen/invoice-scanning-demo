import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { ScanSessionProvider } from '@/scan/ScanSession';
import { colors } from '@/theme';

export default function RootLayout() {
  return (
    <ScanSessionProvider>
      <StatusBar style="light" />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.bg } }} />
    </ScanSessionProvider>
  );
}
