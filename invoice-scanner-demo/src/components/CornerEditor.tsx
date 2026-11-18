import { useRef, useState } from 'react';
import { Image, Pressable, StyleSheet, Text, View, type GestureResponderEvent } from 'react-native';
import Svg, { Circle, Line, Path, Polygon } from 'react-native-svg';
import type { Pt } from '@/lib/api';
import { colors } from '@/theme';

type Props = {
  uri: string;
  photoWidth: number;
  photoHeight: number;
  initial: Pt[]; // starting corners (the automatic suggestion), TL, TR, BR, BL, normalised 0..1
  onBack: () => void;
  onCrop: (corners: Pt[]) => void;
};

const GRAB_RADIUS = 70; // px: how close a touch must be to a corner to grab it
const HANDLE = 16;
const LOUPE = 140;
const ZOOM = 2.5;
const FULL_PHOTO: Pt[] = [
  [0.02, 0.02],
  [0.98, 0.02],
  [0.98, 0.98],
  [0.02, 0.98],
];
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

/**
 * The photo with 4 draggable corners (and a magnifier while dragging), like the crop step of
 * CamScanner-style apps. Starts from the automatic suggestion; the user has the final say.
 */
export function CornerEditor({ uri, photoWidth, photoHeight, initial, onBack, onCrop }: Props) {
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [corners, setCorners] = useState<Pt[]>(initial);
  const [activeIdx, setActiveIdx] = useState<number | null>(null);

  // The photo, fitted inside the stage.
  const scale = box.w && box.h ? Math.min(box.w / photoWidth, box.h / photoHeight) : 0;
  const dw = photoWidth * scale;
  const dh = photoHeight * scale;
  const ox = (box.w - dw) / 2;
  const oy = (box.h - dh) / 2;
  const toScreen = ([x, y]: Pt) => [ox + x * dw, oy + y * dh] as const;

  // Raw responder props: handlers are re-created each render, so they see the latest corners
  // and layout; only the in-progress drag lives in a ref.
  const drag = useRef<{ idx: number; start: Pt; px: number; py: number } | null>(null);
  const endDrag = () => {
    drag.current = null;
    setActiveIdx(null);
  };
  const responder = {
    onStartShouldSetResponder: () => true,
    onMoveShouldSetResponder: () => true,
    onResponderTerminationRequest: () => false,
    onResponderGrant: (e: GestureResponderEvent) => {
      const { locationX, locationY, pageX, pageY } = e.nativeEvent;
      const dists = corners.map((p) => {
        const [cx, cy] = toScreen(p);
        return Math.hypot(cx - locationX, cy - locationY);
      });
      const idx = dists.indexOf(Math.min(...dists));
      if (dists[idx] <= GRAB_RADIUS) {
        drag.current = { idx, start: corners[idx], px: pageX, py: pageY };
        setActiveIdx(idx);
      }
    },
    onResponderMove: (e: GestureResponderEvent) => {
      const d = drag.current;
      if (!d || !dw) return;
      const next: Pt = [
        clamp01(d.start[0] + (e.nativeEvent.pageX - d.px) / dw),
        clamp01(d.start[1] + (e.nativeEvent.pageY - d.py) / dh),
      ];
      setCorners((prev) => prev.map((p, i) => (i === d.idx ? next : p)));
    },
    onResponderRelease: endDrag,
    onResponderTerminate: endDrag,
  };

  const pts = corners.map(toScreen);
  const poly = pts.map((p) => p.join(',')).join(' ');
  // Everything outside the selection, dimmed (even-odd fill: the frame minus the polygon).
  const shade = `M0,0 H${box.w} V${box.h} H0 Z M${pts.map((p) => p.join(',')).join(' L')} Z`;
  const active = activeIdx !== null ? corners[activeIdx] : null;

  return (
    <View style={styles.root}>
      <Text style={styles.hint}>Drag the 4 corners onto the corners of the invoice</Text>
      <View
        style={styles.stage}
        onLayout={(e) => setBox({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}
        {...responder}
      >
        {scale > 0 && (
          <View pointerEvents="none" style={StyleSheet.absoluteFill}>
            <Image source={{ uri }} style={{ position: 'absolute', left: ox, top: oy, width: dw, height: dh }} />
            <Svg width={box.w} height={box.h}>
              <Path d={shade} fill="rgba(0,0,0,0.5)" fillRule="evenodd" />
              <Polygon points={poly} fill="none" stroke={colors.primary} strokeWidth={3} strokeLinejoin="round" />
              {pts.map(([cx, cy], i) => (
                <Circle
                  key={i}
                  cx={cx}
                  cy={cy}
                  r={activeIdx === i ? HANDLE + 6 : HANDLE}
                  fill="rgba(255,255,255,0.3)"
                  stroke={colors.primary}
                  strokeWidth={3}
                />
              ))}
              {pts.map(([cx, cy], i) => (
                <Circle key={`dot${i}`} cx={cx} cy={cy} r={3} fill="#fff" />
              ))}
            </Svg>
            {active && (
              <View style={[styles.loupe, active[0] < 0.5 ? { right: 12 } : { left: 12 }]}>
                <Image
                  source={{ uri }}
                  style={{
                    position: 'absolute',
                    width: dw * ZOOM,
                    height: dh * ZOOM,
                    left: LOUPE / 2 - active[0] * dw * ZOOM,
                    top: LOUPE / 2 - active[1] * dh * ZOOM,
                  }}
                />
                <Svg width={LOUPE} height={LOUPE} style={StyleSheet.absoluteFill}>
                  <Line x1={LOUPE / 2} y1={0} x2={LOUPE / 2} y2={LOUPE} stroke={colors.primary} strokeWidth={1.5} />
                  <Line x1={0} y1={LOUPE / 2} x2={LOUPE} y2={LOUPE / 2} stroke={colors.primary} strokeWidth={1.5} />
                </Svg>
              </View>
            )}
          </View>
        )}
      </View>
      <View style={styles.bar}>
        <Pressable style={[styles.btn, styles.btnGhost]} onPress={onBack}>
          <Text style={styles.btnGhostText}>Back</Text>
        </Pressable>
        <Pressable style={[styles.btn, styles.btnGhost]} onPress={() => setCorners(FULL_PHOTO)}>
          <Text style={styles.btnGhostText}>Full photo</Text>
        </Pressable>
        <Pressable style={[styles.btn, styles.btnPrimary]} onPress={() => onCrop(corners)}>
          <Text style={styles.btnPrimaryText}>Crop</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  hint: { color: '#ddd', textAlign: 'center', paddingVertical: 10, fontSize: 14 },
  stage: { flex: 1, margin: 16 },
  loupe: {
    position: 'absolute',
    top: 12,
    width: LOUPE,
    height: LOUPE,
    borderRadius: LOUPE / 2,
    overflow: 'hidden',
    borderWidth: 3,
    borderColor: '#fff',
    backgroundColor: '#000',
  },
  bar: { flexDirection: 'row', gap: 10, padding: 16 },
  btn: { flex: 1, paddingVertical: 14, borderRadius: 12, alignItems: 'center' },
  btnGhost: { backgroundColor: '#2a2a2a' },
  btnGhostText: { color: colors.text, fontWeight: '600' },
  btnPrimary: { backgroundColor: colors.primary },
  btnPrimaryText: { color: colors.text, fontWeight: '700' },
});
