import { PCDLoader } from 'three/examples/jsm/loaders/PCDLoader';
import { compactCloud } from './pointcloud-budget';

onmessage = ({ data }): void => {
    try {
        const started = performance.now();
        const cloud = new PCDLoader().parse(data.buffer);
        const attributes = Object.fromEntries(Object.entries(cloud.geometry.attributes).map(([name, value]) =>
            [name, { array: value.array, itemSize: value.itemSize }]));
        const compact = compactCloud(attributes, data.maximum, data.focus);
        const sizes = Object.fromEntries(Object.entries(cloud.geometry.attributes).map(([name, value]) => [name, value.itemSize]));
        cloud.geometry.dispose(); (cloud.material as any).dispose();
        postMessage({ ...compact, sizes, parse_ms: performance.now() - started },
            [...Object.values(compact.attributes).map((a) => a.buffer), compact.indices.buffer]);
    } catch { postMessage({ error: 'Point cloud decode failed' }); }
};
