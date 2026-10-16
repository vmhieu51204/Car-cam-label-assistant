"""Generate validated SAM3 demos for offline playback. Use --env-file for inference credentials."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
from uuid import NAMESPACE_URL, uuid5
from PIL import Image, ImageDraw, ImageFont
from kaggle.checkpoint_contract import checkpoint_identity, model_identity, validate_checkpoint
from src.config import Settings
from src.models.video import FrameManifest, VideoJob, VideoJobStatus
from src.services.video.chunks import plan_chunks
from src.services.video.extractor import FFmpegVideoExtractor
from src.services.video.kaggle_batch import KaggleBatchError, KaggleBatchRunner
from src.services.video.sequence_kaggle import make_track_set, parse_sequence_result, sequence_payload
from src.services.video.workspace import VideoWorkspace

ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.RLock()


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def trust_system_certificates():
    # Add the computer's existing trusted roots; HTTPS verification stays enabled.
    if os.name == 'nt' and not os.environ.get('REQUESTS_CA_BUNDLE'):
        import certifi
        certificates = {ssl.DER_cert_to_PEM_cert(c) for c in ssl.create_default_context().get_ca_certs(binary_form=True)}
        bundle = Path(tempfile.gettempdir()) / 'car-cam-system-ca.pem'
        bundle.write_text(Path(certifi.where()).read_text() + '\n' + ''.join(certificates), encoding='ascii')
        os.environ['REQUESTS_CA_BUNDLE'] = str(bundle)


class DemoKaggleRunner(KaggleBatchRunner):
    def _wait_for_dataset(self, reference):
        # Some CLI versions return 404 from status for readable private datasets.
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                result = self._command('datasets', 'files', reference, timeout=60)
                if 'payload/' in result.stdout or 'frames.json' in result.stdout or 'payload.zip' in result.stdout:
                    return
            except KaggleBatchError:
                pass
            time.sleep(min(self.poll_seconds, max(0, deadline-time.monotonic())))
        raise KaggleBatchError('Uploaded demo dataset did not become readable')


def render_video(manifest, tracks, frames, source_video, destination):
    by_frame = {}
    for track in tracks.tracks:
        for obs in track.observations:
            if not obs.outside:
                by_frame.setdefault(obs.frame_index, []).append((track, obs))
    try: font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    except OSError: font = ImageFont.load_default(size=16)
    with tempfile.TemporaryDirectory(prefix='car-cam-render-') as temporary:
        temp = Path(temporary)
        for frame in manifest.frames:
            with Image.open(frames / Path(frame.relative_path).name) as original:
                image = original.convert('RGB')
            draw = ImageDraw.Draw(image)
            for track, obs in by_frame.get(frame.frame_index, []):
                box = (obs.bbox.xtl, obs.bbox.ytl, obs.bbox.xbr, obs.bbox.ybr)
                color = tuple(70 + n % 170 for n in hashlib.sha256(track.track_id.encode()).digest()[:3])
                draw.rectangle(box, outline=color, width=3)
                label = track.label_name + ' ' + track.track_id.rsplit(':', 1)[-1]
                x, y = box[0], max(0, box[1]-23)
                right = min(image.width, x + draw.textlength(label, font=font)+10)
                draw.rectangle((x,y,right,y+23), fill=color)
                draw.text((x+4,y+2), label, fill='black', font=font)
            draw.rectangle((8,8,320,33), fill='#172b42')
            draw.text((14,11), f'SAM3 draft | Frame {frame.frame_index}', fill='white', font=font)
            image.save(temp / f'frame_{frame.frame_index:06d}.jpg', quality=94)
        shutil.copy2(temp/'frame_000000.jpg', destination/'poster.jpg')
        output = destination/'annotated.mp4'
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-framerate',str(manifest.source_fps),
            '-i',str(temp/'frame_%06d.jpg'),'-i',str(source_video),'-map','0:v:0','-map','1:a?',
            '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-shortest',
            '-movflags','+faststart',str(output)], check=True, timeout=600)
        probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
            'stream=nb_frames,width,height','-of','json',str(output)],capture_output=True,text=True,check=True)
        if int(json.loads(probe.stdout)['streams'][0]['nb_frames']) != manifest.frame_count:
            raise ValueError('Rendered result does not preserve every source frame')
        return output


def update_catalog(output, clip):
    with LOCK:
        path = output/'catalog.json'
        catalog = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'schema':'car-cam-demo-catalog/v1','videos':[]}
        catalog['videos'] = [row for row in catalog['videos'] if row['id'] != clip['id']] + [clip]
        catalog['videos'].sort(key=lambda row: row['name'])
        catalog['default_video_id'] = max(catalog['videos'], key=lambda row: row['frame_count'])['id']
        write_json(path, catalog)


def process_one(video, settings, command, classes, output):
    source_sha = digest(video)
    notebook = ROOT/'kaggle/notebooks/autolabel_sam3_benchmark.ipynb'
    code_sha = digest(notebook)
    job_id = str(uuid5(NAMESPACE_URL, json.dumps([source_sha, classes, code_sha])))
    clip_id = video.stem
    work = ROOT/'state/demo-processing'
    workspace = VideoWorkspace(work/'jobs')
    directory = workspace.job_dir(job_id, create=True)
    target = output/clip_id; target.mkdir(parents=True,exist_ok=True)
    completed = target/'demo.json'
    if completed.exists():
        saved = json.loads(completed.read_text(encoding='utf-8'))
        if saved.get('source_sha256') == source_sha and saved.get('notebook_sha256') == code_sha and saved.get('classes') == list(classes) and (target/'annotated.mp4').is_file():
            print(f'{video.name}: verified saved demo reused', flush=True)
            update_catalog(output, saved['catalog_entry']); return saved['catalog_entry']
    manifest_path = directory/'manifest.json'
    if manifest_path.exists():
        manifest = FrameManifest.model_validate_json(manifest_path.read_bytes())
        workspace.verify_frame_files(manifest)
    else:
        now = datetime.now(UTC)
        job = VideoJob(job_id=job_id,status=VideoJobStatus.QUEUED,source_name=video.name,source_sha256=source_sha,
            source_size_bytes=video.stat().st_size,created_at=now,updated_at=now,stage='demo-extraction')
        shutil.copy2(video,workspace.source_path(job))
        extracted = FFmpegVideoExtractor(workspace).extract(job)
        manifest = extracted.model_copy(update={'source_fps':extracted.average_fps,'target_fps':extracted.average_fps,
            'source_frame_count':extracted.frame_count,'sampling_strategy':'all_frames','fps_source':'ffprobe',
            'frames':[frame.model_copy(update={'source_frame_index':frame.frame_index}) for frame in extracted.frames]})
        write_json(manifest_path,manifest.model_dump(mode='json',by_alias=True))
    print(f'{video.name}: {manifest.frame_count} frames extracted; submitting private SAM3 inference',flush=True)
    files, config = sequence_payload(manifest,workspace,classes)
    benchmark = json.loads(files['benchmark-config.json']); benchmark['code_sha256'] = code_sha
    files['benchmark-config.json'] = json.dumps(benchmark).encode()
    result_path = directory/'model-result.json'
    if result_path.exists(): result = json.loads(result_path.read_text(encoding='utf-8'))
    else:
        runner = DemoKaggleRunner(username=settings.kaggle_username,api_token=settings.kaggle_api_token.get_secret_value(),
            command=command,work_root=work/'inference',machine_shape=settings.kaggle_machine_shape,
            sam3_checkpoint_dataset=settings.kaggle_sam3_checkpoint_dataset,poll_seconds=15,timeout_s=10800)
        result = runner.run(stage='sam3-sequence',job_id=job_id,files=files,code_file=notebook)
        write_json(result_path,result)
    rows = result.get('configurations',[])
    if len(rows)!=1 or rows[0].get('status')!='completed':
        raise ValueError(f'SAM3 did not complete: {[row.get("error",row.get("status")) for row in rows]}')
    plans = plan_chunks(manifest.frame_count, window_size=config['window_size'], overlap=config['overlap'],
        scene_cut_frames=tuple(config.get('scene_cut_frames',[])))
    entries = rows[0].get('window_checkpoints',[])
    if sorted(entry['window_id'] for entry in entries) != list(range(len(plans))):
        raise ValueError('Missing independently verified model window coverage')
    identity = checkpoint_identity(json.loads(files['frames.json'])['frames'], config, code_sha)
    for entry in entries:
        plan = plans[entry['window_id']]
        validate_checkpoint(entry, identity=identity,window_id=plan.index,frame_ids=list(range(plan.start,plan.stop)),model=model_identity(result['model_runtime']))
        for prediction in entry['predictions']:
            frame = manifest.frames[prediction['frame_index']]
            if prediction['label'] not in classes or prediction['bbox'][2] > frame.width or prediction['bbox'][3] > frame.height:
                raise ValueError('Model box or class lies outside the requested input contract')
    stitched, component = parse_sequence_result(result,expected_config=config,frame_count=manifest.frame_count)
    if not component.checkpoint_sha256: raise ValueError('Model checkpoint SHA-256 missing')
    labels = [{'id':i+1,'name':name} for i,name in enumerate(classes)]
    tracks = make_track_set(manifest=manifest,manifest_sha256=digest(manifest_path),stitched=stitched,
        labels=labels,classes=classes,component=component)
    frames = target/'frames';frames.mkdir(exist_ok=True)
    for frame in manifest.frames:
        shutil.copy2(directory/frame.relative_path,frames/Path(frame.relative_path).name)
    write_json(target/'manifest.json',manifest.model_dump(mode='json',by_alias=True))
    write_json(target/'tracks.json',tracks.model_dump(mode='json',by_alias=True))
    print(f'{video.name}: {len(tracks.tracks)} stitched tracks; rendering every frame',flush=True)
    rendered = render_video(manifest,tracks,frames,video,target)
    clip = {'id':clip_id,'name':video.name,'frame_count':manifest.frame_count,'fps':manifest.source_fps,
        'width':manifest.width,'height':manifest.height,'track_count':len(tracks.tracks),'directory':clip_id,
        'source_sha256':source_sha,'result_sha256':digest(rendered),'data_source':'sam3-kaggle',
        'checkpoint_sha256':component.checkpoint_sha256,'model_version':component.version}
    write_json(completed,{'schema':'car-cam-processed-demo/v1','source_sha256':source_sha,'notebook_sha256':code_sha,
        'classes':list(classes),'reference_available':False,'catalog_entry':clip,'model_runtime':result['model_runtime'],
        'processed_at':datetime.now(UTC).isoformat(),'result_sha256':digest(result_path)})
    update_catalog(output,clip)
    print(f'{video.name}: READY ({manifest.frame_count} frames, {len(tracks.tracks)} tracks)',flush=True)
    return clip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    parser.add_argument('--kaggle-command',default='kaggle')
    parser.add_argument('--input',type=Path,default=ROOT/'demo_vids')
    parser.add_argument('--output',type=Path,default=ROOT/'demo_results')
    parser.add_argument('--classes',default='car,person,bicycle,truck,bus,motorcycle')
    parser.add_argument('--workers',type=int,default=2,choices=(1,2))
    args = parser.parse_args()
    trust_system_certificates()
    settings = Settings(_env_file=args.env_file)
    videos = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.mp4','.mov','.mkv'})
    if not videos: raise SystemExit('No demo videos found')
    classes = tuple(value.strip() for value in args.classes.split(',') if value.strip())
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one,p,settings,args.kaggle_command,classes,args.output):p for p in videos}
        for future in as_completed(futures):
            try: future.result()
            except Exception as error:
                failures.append(futures[future].name)
                print(f'FAILED {futures[future].name}: {type(error).__name__}: {error}',flush=True)
    if failures: raise SystemExit('Unfinished demos: '+', '.join(failures))
    print('All demo videos processed and default catalog saved.',flush=True)


if __name__ == '__main__':
    main()
