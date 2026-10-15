import JSZip from 'jszip';

onmessage = async ({ data }): Promise<void> => {
    try {
        const zip = await JSZip.loadAsync(data.block);
        const names = Object.keys(zip.files).filter((name) => !zip.files[name].dir);
        const frames = [];
        for (const index of data.indexes) {
            if (!Number.isInteger(index) || index < 0 || index >= names.length) throw new Error('Invalid 3D frame index');
            frames.push(await zip.files[names[index]].async('blob'));
        }
        postMessage({ frames });
    } catch { postMessage({ error: 'Selective 3D archive decode failed' }); }
};
