import { Pressable, StyleSheet, Text, View } from 'react-native';
import { colors } from '@/theme';

type Action = { label: string; onPress: () => void; disabled?: boolean };

export function ResultHeader({ title, left, right }: { title: string; left: Action; right?: Action }) {
  return (
    <View style={styles.header}>
      <HeaderButton {...left} />
      <Text style={styles.title}>{title}</Text>
      {right ? <HeaderButton {...right} /> : <View style={styles.spacer} />}
    </View>
  );
}

function HeaderButton({ label, onPress, disabled }: Action) {
  return (
    <Pressable onPress={onPress} hitSlop={12} disabled={disabled} style={styles.button}>
      <Text style={[styles.link, disabled && styles.disabled]}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  title: { color: colors.text, fontSize: 17, fontWeight: '700' },
  button: { minWidth: 70 },
  spacer: { minWidth: 70 },
  link: { color: colors.link, fontSize: 16, fontWeight: '600' },
  disabled: { opacity: 0.4 },
});
