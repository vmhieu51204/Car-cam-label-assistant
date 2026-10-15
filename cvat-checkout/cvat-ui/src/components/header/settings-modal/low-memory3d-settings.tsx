import React, { useState } from 'react';
import Checkbox from 'antd/lib/checkbox';
import Button from 'antd/lib/button';
import { getCore } from 'cvat-core-wrapper';
import { lowMemory3DEnabled, setLowMemory3D, export3DMeasurements } from 'utils/low-memory3d-profile';

export default function LowMemory3DSettings(): JSX.Element {
    const [enabled, update] = useState(lowMemory3DEnabled());
    return <div>
        <Checkbox checked={enabled} onChange={(event) => {
            const value = event.target.checked;
            getCore().config.lowMemory3D = value; update(value); setLowMemory3D(value);
        }}>3D ít RAM (thử nghiệm)</Checkbox>
        <p>Cache 256 MiB: 192 MiB point cloud và 64 MiB ảnh camera, tải frame đang xem và tối đa tám frame kế tiếp.
            LOD chỉ giảm mật độ hiển thị; nhãn và dữ liệu gốc giữ nguyên.
            Frame quá lớn không được giữ trong cache. Chưa nghiệm thu trên máy 8 GB.</p>
        <Button onClick={export3DMeasurements}>Xuất phép đo 3D</Button>
    </div>;
}
