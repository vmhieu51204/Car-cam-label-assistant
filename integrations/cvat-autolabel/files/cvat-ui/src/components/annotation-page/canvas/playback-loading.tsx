import React, { useEffect, useState } from 'react';
import Spin from 'antd/lib/spin';

// Cached frame swaps should not flash a full-window loading overlay.
// Keep the canvas blocked by its existing ready state while buffering.
export default function PlaybackLoading(): JSX.Element | null {
    const [visible, setVisible] = useState(false);
    useEffect(() => {
        const timer = window.setTimeout(() => setVisible(true), 250);
        return () => window.clearTimeout(timer);
    }, []);
    return visible ? (
        <div className='cvat-spinner-container'>
            <Spin className='cvat-spinner' />
        </div>
    ) : null;
}
