// Copyright (C) 2024 AutoLabel SAM3 Integration
//
// Native CVAT React/TypeScript component for SAM3 Function 1:
// Single-object missing recovery, 4 review tabs, authority control, guarded apply & revert.

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { shallowEqual } from 'utils/redux';
import Tabs from 'antd/lib/tabs';
import Button from 'antd/lib/button';
import Select from 'antd/lib/select';
import InputNumber from 'antd/lib/input-number';
import Radio from 'antd/lib/radio';
import Checkbox from 'antd/lib/checkbox';
import Tag from 'antd/lib/tag';
import Card from 'antd/lib/card';
import Space from 'antd/lib/space';
import Typography from 'antd/lib/typography';
import Alert from 'antd/lib/alert';
import Modal from 'antd/lib/modal';
import Progress from 'antd/lib/progress';
import {
    CheckCircleOutlined,
    CloseCircleOutlined,
    RollbackOutlined,
    UnlockOutlined,
    LockFilled,
    PlayCircleOutlined,
    EyeOutlined,
    ReloadOutlined,
} from '@ant-design/icons';

import { changeFrameAsync } from 'actions/annotation-actions';
import { CombinedState } from 'reducers';

const { Text, Title } = Typography;

interface SAM3AssistPanelProps {
    jobInstance?: any;
}

export default function SAM3AssistPanel(props: SAM3AssistPanelProps): JSX.Element {
    const dispatch = useDispatch();
    const currentFrame = useSelector((state: CombinedState) => state.annotation.player.frame.number);
    const jobInstance = props.jobInstance || useSelector((state: CombinedState) => state.annotation.job.instance);

    const cvatJobId = jobInstance?.id;
    const cvatTaskId = jobInstance?.taskId;

    const [activeTab, setActiveTab] = useState<'needs_review' | 'suggestions' | 'accepted' | 'runs'>('needs_review');
    const [capabilities, setCapabilities] = useState<{ sam3_function1_repair: boolean; vlm_box_rescue: boolean }>({
        sam3_function1_repair: false,
        vlm_box_rescue: false,
    });

    // Authority and Review State
    const [authorityScope, setAuthorityScope] = useState<any>(null);
    const [reviewQueue, setReviewQueue] = useState<any[]>([]);
    const [proposals, setProposals] = useState<any[]>([]);
    const [repairRuns, setRepairRuns] = useState<any[]>([]);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);

    // Single-Object Repair Prompt State
    const [selectedLabelId, setSelectedLabelId] = useState<number | null>(null);
    const [promptBox, setPromptBox] = useState<{ xtl: number; ytl: number; xbr: number; ybr: number }>({
        xtl: 100,
        ytl: 100,
        xbr: 300,
        ybr: 300,
    });
    const [frameStart, setFrameStart] = useState<number>(0);
    const [frameEnd, setFrameEnd] = useState<number>(Math.max(0, currentFrame + 20));
    const [direction, setDirection] = useState<'both' | 'forward' | 'backward'>('both');
    const [savePromptBox, setSavePromptBox] = useState<boolean>(true);
    const [activeRunId, setActiveRunId] = useState<string | null>(null);
    const [busy, setBusy] = useState<boolean>(false);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const backendUrl = '/autolabel/api/v1/video/jobs';

    // Fetch capabilities
    useEffect(() => {
        if (!cvatJobId) return;
        fetch(`${backendUrl}/cvat-jobs/${cvatJobId}/capabilities`, { credentials: 'include' })
            .then((r) => r.json())
            .then((data) => setCapabilities(data))
            .catch(() => {});
    }, [cvatJobId]);

    // Fetch job data & repair status
    const refreshData = useCallback(async () => {
        if (!cvatJobId) return;
        try {
            const jobRes = await fetch(`${backendUrl}/cvat-jobs/${cvatJobId}`, { credentials: 'include' });
            if (!jobRes.ok) return;
            const jobData = await jobRes.json();
            setCurrentJobId(jobData.job_id);

            // Fetch review queue
            if (jobData.job_id) {
                const qRes = await fetch(`${backendUrl}/${jobData.job_id}/review-queue`, { credentials: 'include' });
                if (qRes.ok) {
                    const qData = await qRes.json();
                    setReviewQueue(qData.units || []);
                }

                // Fetch authority
                const authRes = await fetch(`${backendUrl}/${jobData.job_id}/authority`, { credentials: 'include' });
                if (authRes.ok) {
                    setAuthorityScope(await authRes.json());
                }

                // Fetch proposals
                const propRes = await fetch(`${backendUrl}/${jobData.job_id}/repair-proposals`, { credentials: 'include' });
                if (propRes.ok) {
                    setProposals(await propRes.json());
                }

                // Fetch runs
                const runsRes = await fetch(`${backendUrl}/${jobData.job_id}/repair-runs`, { credentials: 'include' });
                if (runsRes.ok) {
                    setRepairRuns(await runsRes.json());
                }
            }
        } catch (err: any) {
            setErrorMsg(err.message || 'Failed to refresh SAM3 data');
        }
    }, [cvatJobId]);

    useEffect(() => {
        refreshData();
    }, [refreshData]);

    // Handle submit single-object repair
    const handleSubmitRepair = async () => {
        if (!currentJobId || !selectedLabelId || !jobInstance) return;
        setBusy(true);
        setErrorMsg(null);
        try {
            const objectUuid = crypto.randomUUID();
            const label = jobInstance.labels.find((l: any) => l.id === selectedLabelId);
            const reqBody = {
                request_id: crypto.randomUUID(),
                job_id: currentJobId,
                cvat_job_id: cvatJobId,
                object_uuid: objectUuid,
                label_id: selectedLabelId,
                label_name: label ? label.name : 'object',
                prompt: {
                    prompt_type: 'box',
                    local_frame: currentFrame,
                    cvat_frame: currentFrame,
                    bbox: promptBox,
                    original_width: 1920,
                    original_height: 1080,
                    save_as_annotation: savePromptBox,
                },
                frame_start: frameStart,
                frame_end: frameEnd,
                direction: direction,
                base_annotation_revision: 0,
            };

            const runRes = await fetch(`${backendUrl}/${currentJobId}/repair-runs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify(reqBody),
            });

            if (!runRes.ok) {
                const errData = await runRes.json();
                throw new Error(errData.detail || 'Repair run failed');
            }

            const runData = await runRes.json();
            setActiveRunId(runData.run_id);
            await refreshData();
            setActiveTab('suggestions');
        } catch (err: any) {
            setErrorMsg(err.message || 'Failed to submit repair');
        } finally {
            setBusy(false);
        }
    };

    // Apply repair proposal (mode: apply_suggestion | accept_selected | replace_protected)
    const handleApplyProposal = async (proposalId: string, mode: 'apply_suggestion' | 'accept_selected' | 'replace_protected') => {
        if (!currentJobId) return;
        setBusy(true);
        try {
            const idempotencyKey = crypto.randomUUID();
            const res = await fetch(`${backendUrl}/${currentJobId}/repair-proposals/${proposalId}/apply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    mode,
                    expected_annotation_revision: 0,
                    idempotency_key: idempotencyKey,
                }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to apply proposal');
            }
            await refreshData();
        } catch (err: any) {
            setErrorMsg(err.message || 'Error applying proposal');
        } finally {
            setBusy(false);
        }
    };

    // Authority action: reopen | reject | allow_replacement
    const handleAuthorityAction = async (objectUuid: string, action: 'reopen' | 'reject' | 'allow_replacement', start: number, end: number) => {
        if (!currentJobId) return;
        setBusy(true);
        try {
            const res = await fetch(`${backendUrl}/${currentJobId}/authority/action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    action,
                    object_uuid: objectUuid,
                    frame_start: start,
                    frame_end: end,
                    expected_annotation_revision: 0,
                }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Authority action failed');
            }
            await refreshData();
        } catch (err: any) {
            setErrorMsg(err.message || 'Authority action failed');
        } finally {
            setBusy(false);
        }
    };

    // Revert commit
    const handleRevertCommit = async (commitId: string) => {
        if (!currentJobId) return;
        setBusy(true);
        try {
            const res = await fetch(`${backendUrl}/${currentJobId}/repair-commits/${commitId}/revert`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    expected_annotation_revision: 0,
                }),
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Revert failed');
            }
            await refreshData();
        } catch (err: any) {
            setErrorMsg(err.message || 'Revert failed');
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className='cvat-sam3-assist-panel' style={{ padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <Title level={5} style={{ margin: 0 }}>
                    SAM3 Recovery & Review
                </Title>
                <Button size='small' icon={<ReloadOutlined />} onClick={refreshData} loading={busy} />
            </div>

            {errorMsg && (
                <Alert
                    type='error'
                    message={errorMsg}
                    closable
                    onClose={() => setErrorMsg(null)}
                    style={{ marginBottom: 12 }}
                />
            )}

            {/* Quick Add Missing Object Tool */}
            <Card size='small' title='Add missing object (SAM3)' style={{ marginBottom: 12 }}>
                <Space direction='vertical' style={{ width: '100%' }}>
                    <div>
                        <Text type='secondary'>Label:</Text>
                        <Select
                            style={{ width: '100%', marginTop: 4 }}
                            placeholder='Select object class'
                            value={selectedLabelId}
                            onChange={(val) => setSelectedLabelId(val)}
                        >
                            {(jobInstance?.labels || []).map((l: any) => (
                                <Select.Option key={l.id} value={l.id}>
                                    {l.name}
                                </Select.Option>
                            ))}
                        </Select>
                    </div>

                    <div style={{ display: 'flex', gap: 8 }}>
                        <div style={{ flex: 1 }}>
                            <Text type='secondary'>Start frame:</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={0}
                                value={frameStart}
                                onChange={(v) => setFrameStart(v || 0)}
                            />
                        </div>
                        <div style={{ flex: 1 }}>
                            <Text type='secondary'>Anchor (current):</Text>
                            <InputNumber style={{ width: '100%', marginTop: 4 }} value={currentFrame} disabled />
                        </div>
                        <div style={{ flex: 1 }}>
                            <Text type='secondary'>End frame:</Text>
                            <InputNumber
                                style={{ width: '100%', marginTop: 4 }}
                                min={frameStart}
                                value={frameEnd}
                                onChange={(v) => setFrameEnd(v || frameStart)}
                            />
                        </div>
                    </div>

                    <div>
                        <Text type='secondary'>Propagation:</Text>
                        <Radio.Group
                            style={{ marginTop: 4, display: 'flex' }}
                            value={direction}
                            onChange={(e) => setDirection(e.target.value)}
                        >
                            <Radio.Button value='both'>Both</Radio.Button>
                            <Radio.Button value='forward'>Forward</Radio.Button>
                            <Radio.Button value='backward'>Backward</Radio.Button>
                        </Radio.Group>
                    </div>

                    <Checkbox checked={savePromptBox} onChange={(e) => setSavePromptBox(e.target.checked)}>
                        Save prompt box as confirmed annotation
                    </Checkbox>

                    <Button
                        type='primary'
                        icon={<PlayCircleOutlined />}
                        onClick={handleSubmitRepair}
                        loading={busy}
                        disabled={!selectedLabelId}
                        style={{ width: '100%' }}
                    >
                        Run SAM3 Repair
                    </Button>
                </Space>
            </Card>

            {/* 4 Review Tabs */}
            <Tabs
                activeKey={activeTab}
                onChange={(key) => setActiveTab(key as any)}
                type='card'
                size='small'
                items={[
                    {
                        key: 'needs_review',
                        label: `Needs review (${reviewQueue.length})`,
                        children: (
                            <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                                {reviewQueue.length === 0 ? (
                                    <Text type='secondary'>No items pending review.</Text>
                                ) : (
                                    reviewQueue.map((item, idx) => (
                                        <Card key={idx} size='small' style={{ marginBottom: 8 }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Tag color={item.priority === 'critical' ? 'red' : item.priority === 'high' ? 'orange' : 'blue'}>
                                                    {item.priority}
                                                </Tag>
                                                <Button
                                                    size='small'
                                                    type='link'
                                                    icon={<EyeOutlined />}
                                                    onClick={() => dispatch(changeFrameAsync(item.frame_start || 0))}
                                                >
                                                    Frame {item.frame_start}
                                                </Button>
                                            </div>
                                            <Text style={{ fontSize: 12 }}>{(item.reason_codes || []).join(', ')}</Text>
                                        </Card>
                                    ))
                                )}
                            </div>
                        ),
                    },
                    {
                        key: 'suggestions',
                        label: `Suggestions (${proposals.length})`,
                        children: (
                            <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                                {proposals.length === 0 ? (
                                    <Text type='secondary'>No suggestions available.</Text>
                                ) : (
                                    proposals.map((p) => (
                                        <Card key={p.proposal_id} size='small' style={{ marginBottom: 8 }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                                                <Text strong>{p.label_name || 'Object'}</Text>
                                                <Tag color='gold'>Rev {p.run_revision}</Tag>
                                            </div>
                                            <Text type='secondary' style={{ fontSize: 12 }}>
                                                Frames: {p.coverage?.processed || 0} processed, {p.coverage?.uncertain_gaps || 0} uncertain
                                            </Text>
                                            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
                                                <Button
                                                    size='small'
                                                    onClick={() => handleApplyProposal(p.proposal_id, 'apply_suggestion')}
                                                    loading={busy}
                                                >
                                                    Apply suggestion
                                                </Button>
                                                <Button
                                                    size='small'
                                                    type='primary'
                                                    icon={<CheckCircleOutlined />}
                                                    onClick={() => handleApplyProposal(p.proposal_id, 'accept_selected')}
                                                    loading={busy}
                                                >
                                                    Accept selected
                                                </Button>
                                            </div>
                                        </Card>
                                    ))
                                )}
                            </div>
                        ),
                    },
                    {
                        key: 'accepted',
                        label: `Accepted (${(authorityScope?.intervals || []).filter((i: any) => i.review_status === 'accepted').length})`,
                        children: (
                            <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                                {(authorityScope?.intervals || [])
                                    .filter((i: any) => i.review_status === 'accepted')
                                    .map((interval: any, idx: number) => (
                                        <Card key={idx} size='small' style={{ marginBottom: 8 }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <div>
                                                    <Tag color='green' icon={<LockFilled />}>
                                                        Protected {interval.frame_start} - {interval.frame_end}
                                                    </Tag>
                                                    <Text style={{ fontSize: 11 }} type='secondary'>
                                                        by {interval.last_decision_by || 'human'}
                                                    </Text>
                                                </div>
                                                <Space size='small'>
                                                    <Button
                                                        size='small'
                                                        onClick={() =>
                                                            handleAuthorityAction(
                                                                interval.object_uuid,
                                                                'reopen',
                                                                interval.frame_start,
                                                                interval.frame_end
                                                            )
                                                        }
                                                        loading={busy}
                                                    >
                                                        Reopen
                                                    </Button>
                                                    <Button
                                                        size='small'
                                                        danger
                                                        onClick={() =>
                                                            handleAuthorityAction(
                                                                interval.object_uuid,
                                                                'reject',
                                                                interval.frame_start,
                                                                interval.frame_end
                                                            )
                                                        }
                                                        loading={busy}
                                                    >
                                                        Reject
                                                    </Button>
                                                </Space>
                                            </div>
                                        </Card>
                                    ))}
                            </div>
                        ),
                    },
                    {
                        key: 'runs',
                        label: `Runs (${repairRuns.length})`,
                        children: (
                            <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                                {repairRuns.length === 0 ? (
                                    <Text type='secondary'>No repair runs executed yet.</Text>
                                ) : (
                                    repairRuns.map((r) => (
                                        <Card key={r.run_id} size='small' style={{ marginBottom: 8 }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                <Text code>{r.run_id.slice(0, 8)}</Text>
                                                <Tag color={r.status === 'ready' ? 'green' : r.status === 'running' ? 'blue' : 'default'}>
                                                    {r.status}
                                                </Tag>
                                            </div>
                                            <Text type='secondary' style={{ fontSize: 12 }}>
                                                Interval: {r.request?.frame_start} - {r.request?.frame_end} ({r.request?.direction})
                                            </Text>
                                        </Card>
                                    ))
                                )}
                            </div>
                        ),
                    },
                ]}
            />
        </div>
    );
}
