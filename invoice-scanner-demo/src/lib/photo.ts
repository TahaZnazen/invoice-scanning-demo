import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';

/** Same as the scan service's working size (MAX_SIDE): anything bigger is thrown away there. */
const UPLOAD_MAX_SIDE = 3000;

/**
 * Shrink a camera photo before upload (12 MP, ~4 MB -> ~3000 px, ~1 MB). Decodes straight to
 * the smaller size, so the full image is never held in memory. Returns a new temp file.
 */
export async function shrinkForUpload(uri: string): Promise<string> {
  const context = ImageManipulator.manipulate(uri, { maxWidth: UPLOAD_MAX_SIDE, maxHeight: UPLOAD_MAX_SIDE });
  try {
    const image = await context.renderAsync();
    try {
      return (await image.saveAsync({ format: SaveFormat.JPEG, compress: 0.85 })).uri;
    } finally {
      image.release();
    }
  } finally {
    context.release();
  }
}

export function deleteQuietly(uri: string) {
  try {
    new File(uri).delete();
  } catch {
    // already gone: fine
  }
}
