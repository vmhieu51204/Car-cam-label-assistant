// Bounded raw-frame cache. Does not change point coordinates or annotations.
export class Stale3DRequest extends Error {}

export class ByteFrameCache<T> {
    private rows = new Map<number, { data: T; bytes: number }>();
    public bytes = 0;
    public current: number | null = null;

    constructor(public readonly limit = 192 * 1024 * 1024) {} // Reserve 64 MiB for camera context images.

    get(frame: number): T | undefined {
        const row = this.rows.get(frame);
        if (row) { this.rows.delete(frame); this.rows.set(frame, row); }
        return row?.data;
    }

    put(frame: number, data: T, bytes: number): boolean {
        if (bytes > this.limit) return false; // Oversize active frames may be displayed, never retained in cache.
        this.drop(frame);
        this.rows.set(frame, { data, bytes }); this.bytes += bytes;
        for (const key of this.rows.keys()) {
            if (this.bytes <= this.limit && this.rows.size <= 9) break;
            if (key !== this.current) this.drop(key);
        }
        if (this.bytes > this.limit) this.drop(frame);
        return this.rows.has(frame);
    }

    private drop(frame: number): void {
        const row = this.rows.get(frame);
        if (row) { this.bytes -= row.bytes; this.rows.delete(frame); }
    }

    clear(): void { this.rows.clear(); this.bytes = 0; this.current = null; }
}

export type Chunk3D = { index: number; frames: number[] };
export type Extract3D = (block: ArrayBuffer, indexes: number[], signal: AbortSignal) => Promise<Blob[]>;

export function extract3D(block: ArrayBuffer, indexes: number[], signal: AbortSignal): Promise<Blob[]> {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL('./selective3d.worker', import.meta.url));
        const finish = (): void => { worker.terminate(); signal.removeEventListener('abort', abort); };
        const abort = (): void => { finish(); reject(new Stale3DRequest()); };
        signal.addEventListener('abort', abort, { once: true });
        if (signal.aborted) { abort(); return; }
        worker.onmessage = ({ data }): void => {
            finish();
            if (data.error) reject(new Error(data.error)); else resolve(data.frames);
        };
        worker.onerror = (): void => { finish(); reject(new Error('3D chunk decode failed')); };
        worker.postMessage({ block, indexes }, [block]);
    });
}

export class LowMemory3DFrames {
    public readonly cache = new ByteFrameCache<Blob>();
    private active: AbortController | null = null;
    private generation = 0;
    private request = 0;
    private pending = new Map<number, Promise<Blob>>();
    private queue: Promise<unknown> = Promise.resolve();

    constructor(
        private fetchChunk: (index: number, signal: AbortSignal) => Promise<ArrayBuffer>,
        private locate: (frame: number) => Chunk3D,
        private extract: Extract3D = extract3D,
        private report: (event: object) => void = () => {},
    ) {}

    async load(frame: number, next: number | number[] | null): Promise<Blob> {
        // Moving into a buffered frame must reuse its download, not abort it.
        if (!this.active || (!this.cache.get(frame) && !this.pending.has(frame))) {
            this.active?.abort();
            this.active = new AbortController();
            ++this.generation;
            this.pending.clear();
            this.queue = Promise.resolve();
        }
        const controller = this.active;
        const generation = this.generation;
        const request = ++this.request;
        this.cache.current = frame;
        const started = performance.now();
        const read = async (selected: number[]): Promise<Blob[]> => {
            const chunk = this.locate(selected[0]);
            const network = performance.now();
            const block = await this.fetchChunk(chunk.index, controller.signal);
            if (generation !== this.generation) throw new Stale3DRequest();
            const decode = performance.now();
            const rows = await this.extract(block, selected.map((f) => chunk.frames.indexOf(f)), controller.signal);
            if (generation !== this.generation) throw new Stale3DRequest();
            if (rows.length !== selected.length || rows.some((row) => !(row instanceof Blob))) {
                throw new Error('3D frame mapping differs');
            }
            this.report({ stage: 'download-decode', frame: selected[0], network_ms: decode - network,
                decode_ms: performance.now() - decode });
            return rows;
        };
        const ahead = [...new Set(Array.isArray(next) ? next : next === null ? [] : [next])]
            .filter((number) => number !== frame).slice(0, 8);
        const buffered = ahead.filter((number) => this.cache.get(number) || this.pending.has(number)).length;
        const selected = [frame, ...(buffered < Math.min(4, ahead.length) ? ahead : [])]
            .filter((number) => !this.cache.get(number) && !this.pending.has(number));
        const groups = new Map<number, number[]>();
        selected.forEach((number) => {
            const index = this.locate(number).index;
            groups.set(index, [...(groups.get(index) || []), number]);
        });
        for (const numbers of groups.values()) {
            const task = this.queue.then(async () => {
                if (controller.signal.aborted || generation !== this.generation) throw new Stale3DRequest();
                const blobs = await read(numbers);
                // Insert farthest first so eviction preserves the nearest upcoming frames.
                for (let i = numbers.length - 1; i >= 0; i--) this.cache.put(numbers[i], blobs[i], blobs[i].size);
                return blobs;
            });
            this.queue = task.catch(() => {});
            numbers.forEach((number, index) => {
                const result = task.then((blobs) => blobs[index]);
                this.pending.set(number, result);
                void result.catch(() => {}).finally(() => {
                    if (this.pending.get(number) === result) this.pending.delete(number);
                });
            });
        }
        const blob = this.cache.get(frame) || await this.pending.get(frame);
        if (request !== this.request || generation !== this.generation) throw new Stale3DRequest();
        if (!blob) throw new Error('3D frame was not buffered');
        this.report({ stage: 'frame-ready', frame, total_ms: performance.now() - started,
            cache_bytes: this.cache.bytes, frame_bytes: blob.size, oversize: blob.size > this.cache.limit });
        return blob;
    }

    close(): void {
        ++this.generation; ++this.request; this.active?.abort(); this.active = null;
        this.pending.clear(); this.queue = Promise.resolve(); this.cache.clear();
    }
}
