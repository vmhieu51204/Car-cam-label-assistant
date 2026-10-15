import * as THREE from 'three';

export class CloudRequest {
    private worker: Worker | null = null;
    private reject: ((error: Error) => void) | null = null;
    private generation = 0;

    cancel(): void {
        ++this.generation; this.worker?.terminate(); this.worker = null;
        this.reject?.(new Error('Point cloud request superseded')); this.reject = null;
    }

    async parse(blob: Blob, maximum: number, focus?: object): Promise<{ cloud: THREE.Points; stats: object }> {
        this.cancel();
        const generation = this.generation;
        const buffer = await blob.arrayBuffer();
        if (generation !== this.generation) throw new Error('Point cloud request superseded');
        return new Promise((resolve, reject) => {
            const worker = new Worker(new URL('./pointcloud.worker', import.meta.url));
            this.worker = worker; this.reject = reject;
            const finish = (): void => { worker.terminate(); this.worker = null; this.reject = null; };
            worker.onmessage = ({ data }): void => {
                finish();
                if (generation !== this.generation || data.error || !data.displayedPoints) {
                    reject(new Error(data.error || 'No finite points, or stale point cloud')); return;
                }
                const geometry = new THREE.BufferGeometry();
                for (const [name, value] of Object.entries(data.attributes)) {
                    geometry.setAttribute(name, new THREE.BufferAttribute(value as Float32Array, data.sizes[name]));
                }
                const material = new THREE.PointsMaterial({ size: 0.05, vertexColors: !!data.attributes.color, color: 0xffffff });
                resolve({ cloud: new THREE.Points(geometry, material), stats: {
                    stage: 'pointcloud', input_points: data.inputPoints, displayed_points: data.displayedPoints,
                    parse_ms: data.parse_ms, lod: data.inputPoints > data.displayedPoints,
                } });
            };
            worker.onerror = (): void => { finish(); reject(new Error('Point cloud worker failed')); };
            worker.postMessage({ buffer, maximum, focus }, [buffer]);
        });
    }
}
