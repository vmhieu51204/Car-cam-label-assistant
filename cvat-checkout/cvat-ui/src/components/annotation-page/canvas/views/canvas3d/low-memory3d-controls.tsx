import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import { CombinedState } from 'reducers';
import { Canvas3d } from 'cvat-canvas3d-wrapper';
import { lowMemory3DEnabled } from 'utils/low-memory3d-profile';

type Status = { frame?: number; stage?: string; cache_bytes?: number; displayed_points?: number;
    input_points?: number; lod?: boolean; oversize?: boolean };

export default function LowMemory3DControls(): JSX.Element | null {
    const canvas = useSelector((state: CombinedState) => state.annotation.canvas.instance) as Canvas3d;
    const [enabled, update] = useState(lowMemory3DEnabled());
    const [detail, setDetail] = useState(false);
    const [status, setStatus] = useState<Status>({});
    useEffect(() => {
        canvas.configure({ lowMemory3D: enabled, detailedObject: detail });
    }, [canvas, enabled, detail]);
    useEffect(() => {
        const profile = (event: Event): void => { update((event as CustomEvent).detail); };
        const metrics = (event: Event): void => {
            const row = (event as CustomEvent<Status>).detail;
            if (!['frame-ready', 'pointcloud'].includes(row.stage || '')) return;
            setStatus((previous) => {
                if (row.stage === 'frame-ready' && row.frame !== previous.frame) return { ...row };
                return { ...previous, ...row };
            });
        };
        window.addEventListener('cvat:3d-memory-profile', profile);
        window.addEventListener('cvat:3d-performance', metrics);
        return () => {
            window.removeEventListener('cvat:3d-memory-profile', profile);
            window.removeEventListener('cvat:3d-performance', metrics);
        };
    }, []);
    if (!enabled) return null;
    return <div style={{ position: 'absolute', top: 4, right: 4, zIndex: 1, background: '#fff', padding: 4 }}>
        <span>3D ít RAM · point cloud {Math.round((status.cache_bytes || 0) / 1048576)}/192 MiB
            {status.displayed_points !== undefined ? ` · ${status.displayed_points}/${status.input_points} điểm` : ''}
            {status.lod ? ' · LOD' : ''}{status.oversize ? ' · frame lớn, không cache' : ''}</span>
        <Button size='small' onClick={() => setDetail(!detail)}>{detail ? 'Tổng quan' : 'Chi tiết object đã chọn'}</Button>
    </div>;
}
