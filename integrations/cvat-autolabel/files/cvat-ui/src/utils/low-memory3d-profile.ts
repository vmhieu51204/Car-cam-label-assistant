const KEY = 'cvat.3d.low-memory';
export function releaseContextCopies(images: Record<string, ImageBitmap>): void {
    if ((images as any)[Symbol.for('cvat.3d.context-owner')]) Object.values(images).forEach((image) => image.close());
}
export function lowMemory3DEnabled(): boolean {
    try { return localStorage.getItem(KEY) === 'true'; } catch { return false; }
}

export function setLowMemory3D(enabled: boolean): void {
    try { localStorage.setItem(KEY, String(enabled)); } catch { /* The session setting still works. */ }
    window.dispatchEvent(new CustomEvent('cvat:3d-memory-profile', { detail: enabled }));
}

const rows: object[] = [];
let currentFrame: number | null = null;
let currentJob: number | null = null;
window.addEventListener('cvat:3d-performance', ((event: CustomEvent) => {
    const allowed = ['stage', 'frame', 'job', 'low_memory', 'network_ms', 'decode_ms', 'total_ms', 'cache_bytes', 'frame_bytes',
        'oversize', 'input_points', 'displayed_points', 'parse_ms', 'lod', 'render_ms', 'geometries', 'textures'];
    const row = Object.fromEntries(allowed.filter((key) => key in event.detail).map((key) => [key, event.detail[key]]));
    if (row.stage === 'frame-ready') currentFrame = row.frame;
    if (row.job !== undefined) currentJob = row.job;
    rows.push({ frame: currentFrame, job: currentJob, low_memory: lowMemory3DEnabled(), ...row, elapsed_ms: performance.now() });
    if (rows.length > 1200) rows.shift();
}) as EventListener);

export function export3DMeasurements(): void {
    const performanceMemory = (performance as any).memory;
    const blob = new Blob([JSON.stringify({ schema: 'cvat-3d-profile/v1', low_memory: lowMemory3DEnabled(),
        cache_budget_bytes: 256 * 1024 * 1024, events: rows,
        js_heap_bytes: performanceMemory?.usedJSHeapSize ?? null,
        note: 'JS heap excludes GPU/Blob memory; measure browser process RAM separately. No 8 GB acceptance implied.',
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
    anchor.href = url; anchor.download = 'cvat-3d-measurements.json'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
