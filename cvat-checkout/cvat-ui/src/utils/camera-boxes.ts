// Saved native-camera detector evidence, shared by the six context panels.
export interface CameraBox {
    bbox: number[];
    label: string;
    score: number;
    track_id: string;
    linked_3d_track_ids: string[];
}

export interface CameraBoxes {
    frame: number;
    detector: string;
    cameras: { camera: string; boxes: CameraBox[] }[];
}

const frames = new Map<string, Promise<CameraBoxes | null>>();
const LIMIT = 24;

export function cameraBoxes(job: number, frame: number): Promise<CameraBoxes | null> {
    const key = `${job}:${frame}`;
    if (!frames.has(key)) {
        const promise = fetch(`/autolabel/api/v1/vm/3d/cvat-jobs/${job}/camera-boxes?frame=${frame}`, {
            credentials: 'same-origin',
        }).then(async (response) => {
            if (response.status === 404) return null;
            if (!response.ok) throw new Error(`Camera boxes: HTTP ${response.status}`);
            const data = await response.json() as CameraBoxes;
            if (data.frame !== frame) throw new Error('Camera box frame differs');
            return data;
        }).catch((error) => {
            frames.delete(key);
            throw error;
        });
        frames.set(key, promise);
        if (frames.size > LIMIT) frames.delete(frames.keys().next().value);
    }
    return frames.get(key)!;
}

export function boxesForImage(data: CameraBoxes | null, imageName: string): CameraBox[] {
    const channel = imageName.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, '');
    return data?.cameras.find((camera) => camera.camera === channel)?.boxes || [];
}

export function drawCameraBoxes(context: CanvasRenderingContext2D, boxes: CameraBox[]): void {
    const { width, height } = context.canvas;
    const scale = Math.max(1, width / 800);
    context.save();
    context.lineWidth = 2 * scale;
    context.font = `${12 * scale}px sans-serif`;
    context.textBaseline = 'top';
    for (const box of boxes) {
        if (box.bbox.length !== 4 || !box.bbox.every(Number.isFinite)) continue;
        const [left, top, right, bottom] = box.bbox;
        const x = Math.max(0, Math.min(width, left));
        const y = Math.max(0, Math.min(height, top));
        const w = Math.min(width, right) - x;
        const h = Math.min(height, bottom) - y;
        if (w <= 0 || h <= 0) continue;
        context.strokeStyle = box.linked_3d_track_ids.length ? '#4dff9a' : '#ffd54f';
        context.strokeRect(x, y, w, h);
        const id = box.track_id.split(':').pop();
        const text = `${box.label} #${id} ${Math.round(box.score * 100)}%`;
        const textWidth = context.measureText(text).width + 6 * scale;
        const labelX = Math.max(0, Math.min(x, width - textWidth));
        const labelY = Math.max(0, y - 17 * scale);
        context.fillStyle = '#101820';
        context.fillRect(labelX, labelY, textWidth, 17 * scale);
        context.fillStyle = context.strokeStyle;
        context.fillText(text, labelX + 3 * scale, labelY + scale);
    }
    context.restore();
}
