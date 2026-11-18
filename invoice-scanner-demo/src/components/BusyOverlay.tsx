import type { ReactNode } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { colors } from '@/theme';

export function BusyOverlay({ label, children }: { label: string; children?: ReactNode }) {
  return (
    <View style={styles.overlay}>
      {children}
      <ActivityIndicator size="large" color={colors.text} />
      <Text style={styles.text}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(0,0,0,0.75)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 14,
  },
  text: { color: colors.text, fontSize: 16, fontWeight: '600' },
});
