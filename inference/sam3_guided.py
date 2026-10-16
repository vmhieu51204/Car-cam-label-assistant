"""Single-object SAM3 box/point tracking; same worker for local CUDA and private Kaggle."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SAM3_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SCHEMA = "car-cam-guided-result/v1"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prompt_arguments(prompt, width, height):
    """The pinned tracker expects coordinates relative to the original frame."""
    if prompt['kind'] == 'box':
        import numpy as np
        b = prompt['bbox']
        return {'box': np.asarray([[b['xtl']/width, b['ytl']/height, b['xbr']/width, b['ybr']/height]], dtype='float32')}
    return {'points': [[p['x']/width, p['y']/height] for p in prompt['points']],
            'labels': [p['label'] for p in prompt['points']]}


def frame_passes(count, anchor, direction):
    passes = []
    if direction in ('both', 'forward'):
        passes.append(list(range(anchor, count)))
    if direction in ('both', 'backward'):
        passes.append(list(range(anchor, -1, -1)))
    return passes


def mask_observation(mask, frame, fallback):
    import numpy as np
    array = mask.detach().cpu().numpy() if hasattr(mask, 'detach') else np.asarray(mask)
    array = np.asarray(array).squeeze() > 0
    if array.ndim != 2:
        raise ValueError('SAM3 returned a non-2D object mask')
    pixels = np.argwhere(array)
    box = dict(fallback)
    if pixels.size:
        y, x = pixels.min(axis=0); bottom, right = pixels.max(axis=0)
        box = {'xtl': float(x), 'ytl': float(y), 'xbr': float(right+1), 'ybr': float(bottom+1)}
    return {'frame_index': frame, 'bbox': box, 'mask_area_px': int(len(pixels)),
            'outside': not bool(pixels.size), 'score': None}


def propagate(tracker, request, root, work_root, on_progress=lambda *args: None):
    """Fresh chronological state in each direction, mapped back to original frame IDs."""
    import numpy as np
    records = request['frames']; prompt = request['prompt']; anchor = prompt['frame']
    width, height = request['width'], request['height']
    fallback = prompt.get('bbox') or {'xtl': prompt['points'][0]['x'], 'ytl': prompt['points'][0]['y'],
        'xbr': min(width, prompt['points'][0]['x']+1), 'ybr': min(height, prompt['points'][0]['y']+1)}
    observations = {}
    for pass_number, order in enumerate(frame_passes(len(records), anchor, request['direction'])):
        directory = Path(work_root)/f'pass-{pass_number}'; directory.mkdir(parents=True)
        for local, frame in enumerate(order):
            shutil.copyfile(Path(root)/records[frame]['relative_path'], directory/f'{local:06d}.jpg')
        state = tracker.init_state(video_path=str(directory), offload_video_to_cpu=True, offload_state_to_cpu=True)
        try:
            tracker.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                **prompt_arguments(prompt, width, height))
            seen = set(); previous = fallback
            for local, object_ids, _low, masks, _scores in tracker.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=len(order), reverse=False, propagate_preflight=True):
                if type(local) is not int or local in seen or not 0 <= local < len(order):
                    raise ValueError('SAM3 returned duplicate or out-of-range frame IDs')
                seen.add(local)
                ids = [int(x) for x in object_ids]
                raw = masks[ids.index(1)] if 1 in ids else np.zeros((height,width),dtype=bool)
                observation = mask_observation(raw, order[local], previous)
                if not observation['outside']:
                    previous = observation['bbox']
                # Preserve the human anchor box; all other boxes come from model masks.
                if order[local] == anchor and prompt['kind'] == 'box':
                    observation['bbox'] = dict(prompt['bbox']); observation['outside'] = False
                observations.setdefault(order[local], observation)
                on_progress(len(observations), len(records))
            if seen != set(range(len(order))):
                raise ValueError('SAM3 did not evaluate every requested frame')
        finally:
            del state
            shutil.rmtree(directory)
            gc.collect()
    return [observations[i] for i in sorted(observations)]


def run(request, root, checkpoint, output):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('SAM3 guided tracking requires a CUDA GPU; configure private Kaggle or local CUDA.')
    body = {k:v for k,v in request.items() if k != 'request_sha256'}
    if canonical_sha(body) != request['request_sha256']:
        raise ValueError('Guided request checksum differs')
    for record in request['frames']:
        path = (root/record['relative_path']).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Guided input frame checksum differs')
    with checkpoint.open('rb') as handle:
        checkpoint_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    if checkpoint_sha != request['checkpoint_sha256']:
        raise ValueError('SAM3 checkpoint differs from this video’s saved model')
    from sam3.model_builder import build_sam3_video_model
    print('Loading SAM3 guided tracker', flush=True)
    model = build_sam3_video_model(checkpoint_path=str(checkpoint), load_from_HF=False)
    model.tracker.backbone = model.detector.backbone
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), tempfile.TemporaryDirectory(prefix='sam3-guided-') as temporary:
        observations = propagate(model.tracker, request, root, temporary,
            lambda n,total: print(f'Tracked {n}/{total} source frames',flush=True))
    result = {'schema': SCHEMA, 'stage': 'sam3', 'job_id': request['job_id'], 'status': 'completed',
        'request_sha256': request['request_sha256'], 'source_sha256': request['source_sha256'],
        'evaluated_frames': [o['frame_index'] for o in observations],
        'component': {'component': 'sam3-guided', 'version': 'sam3@'+SAM3_COMMIT,
                      'checkpoint_sha256': checkpoint_sha, 'config_sha256': request['request_sha256']},
        'track': {'track_id': request['track_id'], 'label_id': request['label_id'], 'label_name': request['label_name'],
                  'source': 'human_seed', 'review_status': 'pending', 'needs_review': True,
                  'review_reasons': ['guided_tracking_pending_review'], 'observations': observations}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result),encoding='utf-8')
    print('Guided result saved',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path); parser.add_argument('--root',type=Path)
    parser.add_argument('--checkpoint',type=Path); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.request:
        run(json.loads(args.request.read_text()), args.root, args.checkpoint, args.output)
        return
    # Kaggle: attach exactly one private input payload and the saved SAM3 checkpoint.
    input_root=Path('/kaggle/input'); root=Path('/kaggle/working/guided-input')
    archives=list(input_root.glob('**/payload.zip'))
    extracted=list(input_root.glob('**/request.json'))
    if len(archives)+len(extracted)!=1:
        raise RuntimeError('Expected exactly one guided input dataset')
    if extracted:
        root=extracted[0].parent
    else:
        root.mkdir(exist_ok=True)
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                if not (root/member.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError('Unsafe guided input archive')
            archive.extractall(root)
    checkpoints=list(input_root.glob('**/sam3.pt'))
    if len(checkpoints)!=1:
        raise RuntimeError('Attach exactly one sam3.pt checkpoint')
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps',
        f'git+https://github.com/facebookresearch/sam3.git@{SAM3_COMMIT}',
        'timm>=1.0.17','ftfy==6.1.1','iopath>=0.1.10','portalocker','huggingface_hub'],check=True)
    run(json.loads((root/'request.json').read_text()),root,checkpoints[0],Path('/kaggle/working/result.json'))


if __name__=='__main__': main()
