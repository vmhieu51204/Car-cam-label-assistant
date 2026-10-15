// Preserve edits made while Save awaited the server, including newly created/deleted objects.
const clone = (value: any): any => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const same = (a: any, b: any): boolean => JSON.stringify(a) === JSON.stringify(b);

function withoutServerIDs(value: any): any {
    if (Array.isArray(value)) return value.map(withoutServerIDs);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
        .filter(([key]) => key !== 'id').map(([key, item]) => [key, withoutServerIDs(item)]));
    return value;
}

function applyLocalDelta(before: any, current: any, saved: any, key = ''): any {
    if (same(before, current)) return clone(saved);
    if (current === undefined) return undefined;
    if (before && current && saved && !Array.isArray(before) && typeof before === 'object' &&
        !Array.isArray(current) && typeof current === 'object' && !Array.isArray(saved) && typeof saved === 'object') {
        const result = clone(saved);
        for (const field of new Set([...Object.keys(before), ...Object.keys(current)])) {
            const value = applyLocalDelta(before[field], current[field], saved[field], field);
            if (value === undefined) delete result[field]; else result[field] = value;
        }
        return result;
    }
    const identityKey = key === 'shapes' ? 'frame' : key === 'attributes' ? 'spec_id' : null;
    if (identityKey && [before, current, saved].every(Array.isArray)) {
        const maps = [before, current, saved].map((rows) => Object.fromEntries(rows.map((r) => [r[identityKey], r])));
        return Object.values(applyLocalDelta(...maps as [any, any, any]));
    }
    // A user's new change remains local, even if an earlier conflict choice selected the server value.
    return clone(current);
}

export function reconcileAfterSave(submitted: any, current: any, saved: any, createdIDs: any): any {
    const result = clone(saved);
    for (const kind of ['tracks', 'shapes', 'tags', 'intervals']) {
        const rows = new Map(result[kind].map((row: any) => [row.id, row]));
        const remaining = new Map(current[kind].map((row: any) => [row.clientID, row]));
        const retained = [];
        for (const original of submitted[kind]) {
            const id = createdIDs[kind]?.[original.clientID] ?? original.id;
            const local = remaining.get(original.clientID);
            remaining.delete(original.clientID);
            if (local === undefined) rows.delete(id);
            else if (rows.has(id)) {
                const delta = applyLocalDelta(original, local, rows.get(id));
                delta.id = id;
                rows.set(id, delta);
            } else if (!same(original, local)) {
                // The server accepted a remote deletion before these new edits existed.
                // Keep them locally for the user's next Save; never resurrect on the server here.
                retained.push(withoutServerIDs(local));
            }
        }
        result[kind] = [...rows.values(), ...retained, ...remaining.values()];
    }
    return result;
}
