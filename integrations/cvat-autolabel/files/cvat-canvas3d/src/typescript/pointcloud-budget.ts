// Pure display selection; retained points keep their original coordinates and indices.
export interface CloudAttribute { array: ArrayLike<number>; itemSize: number }
export type CloudAttributes = Record<string, CloudAttribute>;

export function compactCloud(attributes: CloudAttributes, maximum: number, focus?: {
    center: number[]; size: number[]; inverseRotation: number[];
}): { attributes: Record<string, Float32Array>; indices: Uint32Array; inputPoints: number; displayedPoints: number } {
    const position = attributes.position;
    if (!position || position.itemSize !== 3) throw new Error('Point cloud requires XYZ');
    const count = Math.floor(position.array.length / 3);
    const valid = (i: number): boolean => Number.isFinite(position.array[i * 3]) &&
        Number.isFinite(position.array[i * 3 + 1]) && Number.isFinite(position.array[i * 3 + 2]);
    const inside = (i: number): boolean => {
        if (!focus) return false;
        const v = [0, 1, 2].map((a) => position.array[i * 3 + a] - focus.center[a]);
        return [0, 1, 2].every((a) => Math.abs(v[0] * focus.inverseRotation[a * 3] +
            v[1] * focus.inverseRotation[a * 3 + 1] + v[2] * focus.inverseRotation[a * 3 + 2]) <= focus.size[a] / 2 + 0.1);
    };
    const target = Math.max(1, Math.floor(maximum));
    const selected = new Uint32Array(Math.min(count, target));
    let used = 0;
    if (focus) for (let i = 0; i < count && used < target; i++) if (valid(i) && inside(i)) selected[used++] = i;
    const remaining = target - used;
    if (remaining) {
        const stride = Math.max(1, Math.ceil(count / remaining));
        for (let i = 0; i < count && used < target; i += stride) if (valid(i) && !inside(i)) selected[used++] = i;
    }
    const indices = selected.slice(0, used);
    const result: Record<string, Float32Array> = {};
    for (const [name, attribute] of Object.entries(attributes)) {
        const values = new Float32Array(used * attribute.itemSize);
        for (let i = 0; i < used; i++) for (let k = 0; k < attribute.itemSize; k++) {
            const value = attribute.array[indices[i] * attribute.itemSize + k];
            values[i * attribute.itemSize + k] = Number.isFinite(value) ? value : 0;
        }
        result[name] = values;
    }
    return { attributes: result, indices, inputPoints: count, displayedPoints: used };
}
