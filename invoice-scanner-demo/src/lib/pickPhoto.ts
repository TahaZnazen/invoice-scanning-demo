import { Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

/** Pick one photo from the library. Returns its local uri, or null if cancelled/denied. */
export async function pickPhoto(): Promise<string | null> {
  const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!perm.granted) {
    Alert.alert('Permission needed', 'Allow photo access in Settings.');
    return null;
  }
  const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 });
  return res.canceled ? null : (res.assets[0]?.uri ?? null);
}

/** Take one photo with the phone's normal camera (used where the document scanner isn't available). */
export async function takePhoto(): Promise<string | null> {
  const perm = await ImagePicker.requestCameraPermissionsAsync();
  if (!perm.granted) {
    Alert.alert('Permission needed', 'Allow camera access in Settings.');
    return null;
  }
  const res = await ImagePicker.launchCameraAsync({ mediaTypes: ['images'], quality: 0.9 });
  return res.canceled ? null : (res.assets[0]?.uri ?? null);
}
