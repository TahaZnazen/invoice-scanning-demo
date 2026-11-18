import { useState } from 'react';
import { ActivityIndicator, Image, StyleSheet, Text, View } from 'react-native';
import { colors } from '@/theme';

type Props = {
  uri: string;
  width: number;
  /** height / width to use until the image has loaded (e.g. predicted by the server). */
  initialAspect?: number;
};

/** A page image at full width with its real proportions, a spinner while loading. */
export function PageImage({ uri, width, initialAspect = 1.414 }: Props) {
  const [loaded, setLoaded] = useState<{ uri: string; aspect: number } | null>(null);
  const [failedUri, setFailedUri] = useState<string | null>(null);
  const aspect = loaded?.aspect ?? initialAspect; // keeps the previous shape while a new uri loads

  return (
    <View>
      <Image
        source={{ uri }}
        style={{ width, height: width * aspect, backgroundColor: '#fff' }}
        resizeMode="contain"
        onLoad={(e) => {
          const { width: w, height: h } = e.nativeEvent.source;
          setLoaded({ uri, aspect: h / w });
        }}
        onError={() => setFailedUri(uri)}
      />
      {loaded?.uri !== uri && failedUri !== uri && (
        <ActivityIndicator style={StyleSheet.absoluteFill} color={colors.primary} size="large" />
      )}
      {failedUri === uri && <Text style={styles.error}>Could not load this page.</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  error: { color: colors.warn, textAlign: 'center', paddingVertical: 8 },
});
