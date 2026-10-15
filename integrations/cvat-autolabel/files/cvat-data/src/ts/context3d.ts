// Cache owns originals. Each viewport owns copies and releases them on unmount.
export const CONTEXT_OWNER = Symbol.for('cvat.3d.context-owner');
type Images = Record<string, ImageBitmap>;
export function releaseContextCopies(images: Images): void {
    if ((images as any)[CONTEXT_OWNER]) Object.values(images).forEach((image) => image.close());
}

export class Context3DCache {
    private rows = new Map<number, { images: Images; bytes: number }>();
    private queue: Promise<unknown> = Promise.resolve();
    private closed = false;
    public bytes = 0;
    constructor(private readonly limit = 64 * 1024 * 1024) {}

    load(frame: number, decode: () => Promise<Images>, signal?: AbortSignal): Promise<Images> {
        const run = async (): Promise<Images> => {
            if (this.closed || signal?.aborted) throw frame;
            let row = this.rows.get(frame);
            if (!row) {
                const images = await decode();
                row = { images, bytes: Object.values(images).reduce((sum, image) => sum + image.width * image.height * 4, 0) };
            }
            const copies: Images = {};
            try {
                for (const [key, image] of Object.entries(row.images)) copies[key] = await createImageBitmap(image);
                Object.defineProperty(copies, CONTEXT_OWNER, { value: true });
                if (this.closed || signal?.aborted) throw frame;
                if (!this.rows.has(frame) && row.bytes <= this.limit) {
                    this.rows.set(frame, row); this.bytes += row.bytes;
                }
                for (const [key, cached] of this.rows) {
                    if (this.bytes <= this.limit) break;
                    if (key === frame) continue;
                    Object.values(cached.images).forEach((image) => image.close());
                    this.rows.delete(key); this.bytes -= cached.bytes;
                }
                return copies;
            } catch (error) {
                Object.values(copies).forEach((image) => image.close()); throw error;
            } finally {
                if (!this.rows.has(frame)) Object.values(row.images).forEach((image) => image.close());
            }
        };
        const request = this.queue.then(run);
        this.queue = request.catch(() => {});
        return request;
    }

    close(): void {
        this.closed = true;
        // A pending copy must finish before its originals are released.
        void this.queue.finally(() => {
            for (const row of this.rows.values()) Object.values(row.images).forEach((image) => image.close());
            this.rows.clear(); this.bytes = 0;
        });
    }
}
