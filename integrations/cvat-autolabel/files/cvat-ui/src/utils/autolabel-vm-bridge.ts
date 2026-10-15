// CVAT Redux bridge: keep canvas state and never discard unsaved annotations.
import { Job, getCore } from 'cvat-core-wrapper';
import { setLowMemory3D } from 'utils/low-memory3d-profile';
import { activateObject, fetchAnnotationsAsync } from 'actions/annotation-actions';
import { CombinedState } from 'reducers';

export function installAutoLabelBridge(
    job: Job, changeFrame: (frame: number) => void,
    dispatch: (action: any) => any, getState: () => CombinedState,
): () => void {
    const identity = (object: any): string | number => {
        const attribute = object.label.attributes.find((item: any) => item.name === 'autolabel_track_id');
        return attribute && object.attributes[attribute.id] ? object.attributes[attribute.id] : object.serverID;
    };
    let active = true;
    let savedSelection: any = null;
    const geometry = (object: any): string => JSON.stringify([object.label.id, object.frame, object.points]);
    const beforeSave = (event: Event): void => {
        const current = getState().annotation;
        const object = current.annotations.states.find((item) => item.clientID === current.annotations.activatedStateID);
        if (!object) { savedSelection = null; return; }
        const created = (event as CustomEvent).detail?.createdIDs || {};
        savedSelection = { identity: identity(object), serverID: object.serverID || created[`${object.objectType}s`]?.[object.clientID],
            geometry: geometry(object), frame: current.player.frame.number };
    };
    const afterSave = async (): Promise<void> => {
        if (!savedSelection || !active) return;
        const previous = savedSelection;
        await dispatch(fetchAnnotationsAsync());
        if (!active || getState().annotation.player.frame.number !== previous.frame) return;
        const states = getState().annotation.annotations.states;
        const object = states.find((item) => previous.identity && identity(item) === previous.identity) ||
            states.find((item) => previous.serverID && item.serverID === previous.serverID) ||
            states.find((item) => geometry(item) === previous.geometry);
        if (object) dispatch(activateObject(object.clientID, null, null));
    };
    const listener = async (event: Event): Promise<void> => {
        const { requestID, action, frame, trackID } = (event as CustomEvent).detail;
        const reply = (value: object): void => {
            if (active) window.dispatchEvent(new CustomEvent('autolabel:vm-reply', { detail: { requestID, ...value } }));
        };
        try {
            if (action === 'memory3d') {
                const { enabled } = (event as CustomEvent).detail;
                if (typeof enabled !== 'boolean') throw new Error('Invalid 3D display setting');
                getCore().config.lowMemory3D = enabled;
                setLowMemory3D(enabled);
                reply({ ok: true, enabled });
                return;
            }
            if (action === 'status') {
                reply({ ok: true, dirty: job.annotations.hasUnsavedChanges(), frame: getState().annotation.player.frame.number });
                return;
            }
            if (job.annotations.hasUnsavedChanges()) {
                throw new Error('Lưu chỉnh sửa trước khi đồng bộ kết quả mới');
            }
            const current = getState().annotation;
            const selected = current.annotations.states.find((item) => item.clientID === current.annotations.activatedStateID);
            const selectedIdentity = selected ? identity(selected) : null;
            if (action === 'sync') {
                await job.annotations.clear({ reload: true, preserveUnsaved: true });
                await dispatch(fetchAnnotationsAsync());
                const restored = getState().annotation.annotations.states.find((item) => identity(item) === selectedIdentity);
                if (restored) dispatch(activateObject(restored.clientID, null, null));
                reply({ ok: true });
                return;
            }
            if (!Number.isInteger(frame) || frame < job.startFrame || frame > job.stopFrame) {
                throw new Error('Frame nằm ngoài job');
            }
            changeFrame(frame);
            const deadline = Date.now() + 5000;
            while (active && (getState().annotation.player.frame.number !== frame || getState().annotation.player.frame.fetching)) {
                if (Date.now() > deadline) throw new Error('Chưa tải được frame');
                await new Promise((resolve) => window.setTimeout(resolve, 25));
            }
            if (trackID) {
                await dispatch(fetchAnnotationsAsync());
                const object = getState().annotation.annotations.states.find((item) => identity(item) === trackID);
                if (!object) throw new Error('Track không có observation tại frame này');
                dispatch(activateObject(object.clientID, null, null));
            }
            reply({ ok: true, frame });
        } catch (error) {
            reply({ ok: false, error: String(error) });
        }
    };
    window.addEventListener('autolabel:vm-command', listener);
    window.addEventListener('autolabel:save-before', beforeSave);
    window.addEventListener('autolabel:save-after', afterSave);
    return () => {
        active = false;
        window.removeEventListener('autolabel:vm-command', listener);
        window.removeEventListener('autolabel:save-before', beforeSave);
        window.removeEventListener('autolabel:save-after', afterSave);
    };
}
