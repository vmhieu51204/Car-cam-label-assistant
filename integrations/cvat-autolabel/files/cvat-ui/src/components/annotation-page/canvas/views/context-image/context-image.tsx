// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import './styles.scss';
import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import { shallowEqual } from 'utils/redux';
import PropTypes from 'prop-types';
import notification from 'antd/lib/notification';
import Spin from 'antd/lib/spin';
import Text from 'antd/lib/typography/Text';
import { SettingOutlined } from '@ant-design/icons';

import CVATTooltop from 'components/common/cvat-tooltip';
import { CombinedState } from 'reducers';
import ContextImageSelector from './context-image-selector';
import { releaseContextCopies } from 'utils/low-memory3d-profile';
import { CameraBoxes, cameraBoxes, boxesForImage, drawCameraBoxes } from 'utils/camera-boxes';

interface Props {
    offset: number[];
}

function ContextImage(props: Props): JSX.Element {
    const { offset } = props;
    const defaultFrameOffset = (offset[0] || 0);
    const defaultContextImageOffset = (offset[1] || 0);

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const {
        job,
        frame,
        relatedFiles,
    } = useSelector((state: CombinedState) => ({
        job: state.annotation.job.instance!,
        frame: state.annotation.player.frame.number,
        relatedFiles: state.annotation.player.frame.relatedFiles,
    }), shallowEqual);
    const frameIndex = frame + defaultFrameOffset;

    const [contextImageData, setContextImageData] = useState<Record<string, ImageBitmap>>({});
    const [imageIdentity, setImageIdentity] = useState('');
    const [fetching, setFetching] = useState<boolean>(false);
    const [contextImageOffset, setContextImageOffset] = useState<number>(
        Math.min(defaultContextImageOffset, relatedFiles),
    );

    const [hasError, setHasError] = useState<boolean>(false);
    const [showSelector, setShowSelector] = useState<boolean>(false);
    const [detections, setDetections] = useState<{ frame: number; data: CameraBoxes | null } | null>(null);
    const [boxError, setBoxError] = useState(false);
    const [showBoxes, setShowBoxes] = useState(true);

    useEffect(() => {
        let unmounted = false;
        setBoxError(false);
        cameraBoxes(job.id, frameIndex).then((data) => {
            if (!unmounted) setDetections({ frame: frameIndex, data });
            if (data && frameIndex < job.stopFrame) {
                void cameraBoxes(job.id, frameIndex + 1).catch(() => undefined);
            }
        }).catch(() => {
            if (!unmounted) { setDetections(null); setBoxError(true); }
        });
        return () => { unmounted = true; };
    }, [job.id, frameIndex]);

    useEffect(() => {
        let unmounted = false;
        let owned: Record<string, ImageBitmap> = {};
        const controller = new AbortController();
        const promise = job.frames.contextImage(frameIndex, controller.signal);
        setFetching(true);
        setHasError(false);
        promise.then((imageBitmaps: Record<string, ImageBitmap>) => {
            owned = imageBitmaps;
            if (unmounted) releaseContextCopies(owned);
            if (!unmounted) {
                setContextImageData(imageBitmaps);
                setImageIdentity(`${job.id}:${frameIndex}`);
            }
        }).catch((error: any) => {
            if (!unmounted) {
                setHasError(true);
                notification.error({
                    message: `Could not fetch context images. Frame: ${frameIndex}`,
                    description: error.toString(),
                });
            }
        }).finally(() => {
            if (!unmounted) {
                setFetching(false);
            }
        });

        return () => {
            setContextImageData({});
            unmounted = true;
            controller.abort();
            releaseContextCopies(owned);
        };
    }, [job.id, frameIndex]);

    useEffect(() => {
        if (canvasRef.current) {
            const sortedKeys = Object.keys(contextImageData).sort();
            const key = sortedKeys[contextImageOffset];
            const image = contextImageData[key];
            const context = canvasRef.current.getContext('2d');
            if (context && image && imageIdentity === `${job.id}:${frameIndex}`) {
                canvasRef.current.width = image.width;
                canvasRef.current.height = image.height;
                context.drawImage(image, 0, 0);
                if (showBoxes && detections?.frame === frameIndex) {
                    drawCameraBoxes(context, boxesForImage(detections.data, key));
                }
            }
        }
    }, [contextImageData, contextImageOffset, showBoxes, detections, frameIndex, imageIdentity, job.id]);

    const contextImageName = Object.keys(contextImageData).sort()[contextImageOffset];
    return (
        <div className='cvat-context-image-wrapper'>
            <div className='cvat-context-image-header'>
                { relatedFiles > 1 && (
                    <SettingOutlined
                        className='cvat-context-image-setup-button'
                        onClick={() => {
                            setShowSelector(true);
                        }}
                    />
                )}
                <div className='cvat-context-image-title'>
                    <CVATTooltop title={contextImageName}>
                        <Text>{contextImageName}</Text>
                    </CVATTooltop>
                </div>
            </div>
            <div className='cvat-context-image-boxes'>
                <label>
                    <input type='checkbox' checked={showBoxes} onChange={(event) => setShowBoxes(event.target.checked)} />
                    {' 2D boxes · '}
                </label>
                {boxError ? 'Could not load detections' : detections?.frame === frameIndex && detections.data ? (
                    <span title='Green: linked to a 3D track. Yellow: no unambiguous 3D match. Saved detector predictions; not editable here.'>
                        {`YOLO26s · ${boxesForImage(detections.data, contextImageName || '').length} boxes · green: 3D linked`}
                    </span>
                ) : detections?.frame === frameIndex ? 'No saved 2D detections' : 'Loading detections…'}
            </div>
            { (hasError ||
                (!fetching && contextImageOffset >= Object.keys(contextImageData).length)) && <Text> No data </Text>}
            { fetching && <Spin size='small' /> }
            {
                contextImageOffset < Object.keys(contextImageData).length &&
                <canvas ref={canvasRef} />
            }
            { showSelector && (
                <ContextImageSelector
                    images={contextImageData}
                    offset={contextImageOffset}
                    onChangeOffset={(newContextImageOffset: number) => {
                        setContextImageOffset(newContextImageOffset);
                    }}
                    onClose={() => {
                        setShowSelector(false);
                    }}
                />
            )}
        </div>
    );
}

ContextImage.PropType = {
    offset: PropTypes.arrayOf(PropTypes.number),
};

export default React.memo(ContextImage);
