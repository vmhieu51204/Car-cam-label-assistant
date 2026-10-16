"""Persistent track review and explicit asynchronous SAM3 repair proposals."""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Literal
from uuid import uuid4
from fastapi import HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.models.video import BoundingBox2D, ComponentVersion, TrackSet, VideoTrack
from inference.sam3_guided import SAM3_COMMIT, SCHEMA, canonical_sha, frame_passes


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')


class Point(Strict):
    x: float=Field(ge=0,allow_inf_nan=False)
    y: float=Field(ge=0,allow_inf_nan=False)
    label: Literal[0,1]=1


class Prompt(Strict):
    kind: Literal['box','points']
    frame: int=Field(ge=0)
    bbox: BoundingBox2D | None=None
    points: list[Point]=Field(default_factory=list,max_length=16)

    @model_validator(mode='after')
    def one_prompt(self):
        if self.kind=='box' and (self.bbox is None or self.points):
            raise ValueError('A box prompt requires one box and no points')
        if self.kind=='points' and (self.bbox is not None or not self.points or not any(p.label==1 for p in self.points)):
            raise ValueError('Point prompts require a positive point and no box')
        return self


class Revision(Strict):
    expected_revision: int=Field(ge=0)


class RepairRequest(Revision):
    prompt: Prompt
    label_name: str=Field(min_length=1,max_length=128)
    replace_track_id: str | None=Field(default=None,max_length=128)
    direction: Literal['both','forward','backward']='both'


def atomic_json(path, body):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(body,indent=2),encoding='utf-8')
    temporary.replace(path)


def trust_roots():
    if os.name=='nt' and not os.environ.get('REQUESTS_CA_BUNDLE'):
        import certifi
        pem=Path(tempfile.gettempdir())/'car-cam-system-ca.pem'
        pem.write_text(Path(certifi.where()).read_text()+'\n'+''.join({ssl.DER_cert_to_PEM_cert(c)
            for c in ssl.create_default_context().get_ca_certs(binary_form=True)}),encoding='ascii')
        os.environ['REQUESTS_CA_BUNDLE']=str(pem)


class GuidedWorkspace:
    def __init__(self, session, root, *, state_root=None, runner=None):
        self.session=session
        self.root=Path(root)
        self.state_root=Path(state_root or self.root/'state/review-workspace')
        self.runner=runner
        self.runs={}
        self.undo_stack=[]
        self.revision=0
        self.generation=str(uuid4())

    def capabilities(self):
        provider=('test' if self.runner else 'local' if os.environ.get('SAM3_CHECKPOINT') else 'kaggle'
            if all(os.environ.get(k) for k in ('KAGGLE_USERNAME','KAGGLE_API_TOKEN','KAGGLE_SAM3_CHECKPOINT_DATASET')) else None)
        return {'provider':provider,'configured':provider is not None,'prompts':['box','points'],
            'direction':['both','forward','backward'],'max_frames':300,
            'message':('Private Kaggle GPU; uploads only this clip and your prompt when you run tracking.' if provider=='kaggle'
                else 'Local CUDA SAM3 tracker.' if provider=='local'
                else 'Configure SAM3_CHECKPOINT for local CUDA or Kaggle credentials to run real tracking.')}

    def path(self):
        return self.state_root/(self.session.clip['id']+'.json') if self.session.clip else None

    def restore(self):
        self.generation=str(uuid4())
        self.undo_stack=[]
        self.revision=0
        path=self.path()
        if path and path.is_file():
            saved=json.loads(path.read_text(encoding='utf-8'))
            if saved.get('baseline_sha256')==self.session.clip['result_sha256']:
                tracks=TrackSet.model_validate(saved['tracks'])
                if tracks.job_id!=self.session.manifest.job_id:
                    raise ValueError('Saved edits belong to another video')
                self.session.tracks=tracks
                self.session.states=saved.get('review_states',{})
                self.revision=saved['revision']
        if self.session.clip:self.recover_runs()

    def recover_runs(self):
        for path in (self.state_root/'runs').glob('*/request.json'):
            request=json.loads(path.read_text(encoding='utf-8'))
            status_path=path.parent/'status.json'
            if status_path.is_file():
                run=json.loads(status_path.read_text(encoding='utf-8'))
            elif request.get('source_sha256')==self.session.manifest.source_sha256:
                run={'id':request['job_id'],'base_revision':self.revision,'video_id':self.session.clip['id'],
                    'replace_track_id':request['track_id'] if any(t.track_id==request['track_id'] for t in self.session.tracks.tracks) else None,
                    'prompt':request['prompt'],'label_name':request['label_name'],'direction':request['direction'],
                    'frame_count':len(request['frames']),'provider':self.capabilities()['provider'],'status':'interrupted'}
            else:continue
            if run.get('video_id')!=self.session.clip['id'] or run['id'] in self.runs:continue
            run.setdefault('created_at',path.stat().st_mtime)
            run['generation']=self.generation
            if run['status'] in {'queued','running'}:run['status']='interrupted'
            result_path=path.parent/'result.json'
            if run['status']=='ready' and result_path.is_file():
                result=json.loads(result_path.read_text(encoding='utf-8'))
                self.validate_result(result,request)
                run['result']=result
            self.runs[run['id']]=run

    def persist(self):
        path=self.path()
        if path:
            atomic_json(path,{'baseline_sha256':self.session.clip['result_sha256'],'revision':self.revision,
                'tracks':self.session.tracks.model_dump(mode='json'),'review_states':self.session.states})

    def check(self, revision):
        self.session.require()
        if revision!=self.revision:raise HTTPException(409,'Annotations changed. Refresh before applying this action.')

    def snapshot(self):
        self.undo_stack.append((self.session.tracks.model_copy(deep=True),dict(self.session.states)))
        self.undo_stack=self.undo_stack[-20:]

    def changed(self):
        self.revision+=1
        report=self.session.evaluate(self.session.threshold)
        self.persist()
        return {'revision':self.revision,'report':report}

    def delete(self, track_id, revision):
        self.check(revision)
        self.session.track(track_id)
        self.snapshot()
        self.session.tracks.tracks=[t for t in self.session.tracks.tracks if t.track_id!=track_id]
        self.session.states.pop(track_id,None)
        return self.changed()

    def undo(self, revision):
        self.check(revision)
        if not self.undo_stack:raise HTTPException(409,'No track change to undo in this session')
        self.session.tracks,self.session.states=self.undo_stack.pop()
        return self.changed()

    def labels(self):
        return ['car','person','bicycle','truck','bus','motorcycle'] if self.session.clip else ['car','person']

    def public(self, run):
        return {k:v for k,v in run.items() if k not in {'generation','result','request'}}

    def submit(self, payload):
        self.check(payload.expected_revision)
        if not self.capabilities()['configured']:raise HTTPException(503,self.capabilities()['message'])
        if not self.session.clip:raise HTTPException(409,'Real SAM3 reprocessing requires a recorded video. Select your saved demo clip.')
        if any(r['status'] in {'queued','running'} for r in self.runs.values()):raise HTTPException(409,'One SAM3 repair is already running. Wait for it to finish.')
        manifest=self.session.manifest
        if manifest.frame_count>300 or payload.prompt.frame>=manifest.frame_count:raise HTTPException(422,'Prompt frame is outside the video or the video exceeds 300 frames')
        if payload.label_name not in self.labels():raise HTTPException(422,'Select a known object label')
        prompt=payload.prompt
        if prompt.bbox and (prompt.bbox.xbr>manifest.width or prompt.bbox.ybr>manifest.height):raise HTTPException(422,'Prompt box must fit inside the image')
        if any(pt.x>=manifest.width or pt.y>=manifest.height for pt in prompt.points):raise HTTPException(422,'Prompt point must fit inside the image')
        if payload.replace_track_id:self.session.track(payload.replace_track_id)
        run_id=str(uuid4())
        track_id=payload.replace_track_id or 'guided:'+run_id
        label_ids={t.label_name:t.label_id for t in self.session.tracks.tracks}
        label_id=label_ids.get(payload.label_name)
        if label_id is None:
            existing={t.label_id for t in self.session.tracks.tracks}
            label_id=next(i for i in range(1,102) if i not in existing)
        request={'job_id':run_id,'source_sha256':manifest.source_sha256,'checkpoint_sha256':self.session.clip['checkpoint_sha256'],
            'width':manifest.width,'height':manifest.height,'prompt':prompt.model_dump(exclude_none=True),
            'direction':payload.direction,'track_id':track_id,'label_name':payload.label_name,'label_id':label_id,
            'frames':[{'frame_index':f.frame_index,'relative_path':f.relative_path,'sha256':f.sha256} for f in manifest.frames]}
        request['request_sha256']=canonical_sha(request)
        directory=self.state_root/'runs'/run_id
        directory.mkdir(parents=True)
        atomic_json(directory/'request.json',request)
        run={'id':run_id,'status':'queued','base_revision':self.revision,'generation':self.generation,
            'video_id':self.session.clip['id'],'replace_track_id':payload.replace_track_id,'prompt':prompt.model_dump(exclude_none=True),
            'direction':payload.direction,'label_name':payload.label_name,'frame_count':manifest.frame_count,
            'provider':self.capabilities()['provider'],'created_at':time.time()}
        self.runs[run_id]=run
        atomic_json(directory/'status.json',self.public(run))
        threading.Thread(target=self._run,args=(run_id,request,directory),daemon=True).start()
        return self.public(run)

    def resume(self, run_id, revision):
        self.check(revision)
        run=self.runs.get(run_id)
        if not run or run['status']!='interrupted':raise HTTPException(409,'Only an interrupted run can be resumed')
        if run['base_revision']!=self.revision:raise HTTPException(409,'Annotations changed since this run started')
        if any(r['status'] in {'queued','running'} for r in self.runs.values()):raise HTTPException(409,'A tracking run is already active')
        directory=self.state_root/'runs'/run_id
        request=json.loads((directory/'request.json').read_text(encoding='utf-8'))
        run['status']='queued'
        run['generation']=self.generation
        atomic_json(directory/'status.json',self.public(run))
        threading.Thread(target=self._run,args=(run_id,request,directory),daemon=True).start()
        return self.public(run)

    def _run(self, run_id, request, directory):
        run=self.runs[run_id]
        try:
            with self.session.lock:run['status']='running'
            result=self.runner(request,directory) if self.runner else self.execute(request,directory,run['video_id'],run['provider'])
            track=self.validate_result(result,request)
            with self.session.lock:
                run['result']=result
                run['status']='ready'
                run['visible_frames']=sum(not o.outside for o in track.observations)
            atomic_json(directory/'result.json',result)
        except Exception as error:
            message=str(error)
            token=os.environ.get('KAGGLE_API_TOKEN','')
            if token:message=message.replace(token,'[redacted]')
            with self.session.lock:
                run['status']='failed'
                run['error']=message[:800]
        finally:atomic_json(directory/'status.json',self.public(run))

    def execute(self, request, directory, video_id, provider):
        if provider=='kaggle':
            from src.services.video.kaggle_batch import KaggleBatchRunner,KaggleBatchError
            class Runner(KaggleBatchRunner):
                def _wait_for_dataset(self, reference):
                    deadline=time.monotonic()+600
                    while time.monotonic()<deadline:
                        try:
                            response=self._command('datasets','files',reference,timeout=60)
                            if any(marker in response.stdout for marker in ('frames/frame_','payload/','payload.zip','request.json')):return
                        except KaggleBatchError:pass
                        time.sleep(5)
                    raise KaggleBatchError('Private guided dataset is not ready')
            trust_roots()
            files={'request.json':json.dumps(request).encode()}
            for record in request['frames']:
                path=self.session.library.file(video_id,record['relative_path'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Source frame checksum changed')
                files[record['relative_path']]=path
            bundled_cli=Path(sys.executable).with_name('kaggle.exe' if os.name=='nt' else 'kaggle')
            runner=Runner(username=os.environ['KAGGLE_USERNAME'],api_token=os.environ['KAGGLE_API_TOKEN'],
                work_root=self.state_root/'gpu',command=os.environ.get('KAGGLE_COMMAND') or (str(bundled_cli) if bundled_cli.is_file() else 'kaggle'),
                sam3_checkpoint_dataset=os.environ['KAGGLE_SAM3_CHECKPOINT_DATASET'],
                machine_shape=os.environ.get('KAGGLE_MACHINE_SHAPE','NvidiaTeslaT4'),poll_seconds=10,timeout_s=1800)
            for metadata in sorted((runner.work_root/request['job_id']/'kaggle').glob('*/kernel/kernel-metadata.json'),reverse=True):
                prepared=metadata.parent.parent
                archive_path=prepared/'dataset/payload.zip'
                if not archive_path.is_file():continue
                with zipfile.ZipFile(archive_path) as archive:saved_request=json.loads(archive.read('request.json'))
                if saved_request!=request or (metadata.parent/'main.py').read_bytes()!=(self.root/'inference/sam3_guided.py').read_bytes():continue
                kernel=json.loads(metadata.read_text(encoding='utf-8'))
                reference=kernel['id']
                if not reference.startswith(runner.username+'/autolabel-sam3-') or not kernel.get('is_private'):continue
                submitted=prepared/'kernel-submitted.json'
                if not submitted.is_file():
                    runner._wait_for_dataset(kernel['dataset_sources'][0])
                    response=runner._command('kernels','push','-p',str(metadata.parent),timeout=900)
                    match=re.search(r'Kernel version (\d+)',response.stdout)
                    atomic_json(submitted,{'kernel_reference':reference,'push_version_number':int(match[1]) if match else None})
                return runner._collect_result('sam3',request['job_id'],reference,prepared/'output')
            return runner.run(stage='sam3',job_id=request['job_id'],files=files,
                code_file=self.root/'inference/sam3_guided.py',allow_retry_on_terminal_failure=False)
        (directory/'frames').mkdir(exist_ok=True)
        for record in request['frames']:
            shutil.copyfile(self.session.library.file(video_id,record['relative_path']),directory/record['relative_path'])
        output=directory/'result.json'
        with (directory/'worker.log').open('w',encoding='utf-8') as log:
            subprocess.run([os.environ.get('SAM3_PYTHON',sys.executable),str(self.root/'inference/sam3_guided.py'),
                '--request',str(directory/'request.json'),'--root',str(directory),'--checkpoint',os.environ['SAM3_CHECKPOINT'],
                '--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        return json.loads(output.read_text(encoding='utf-8'))

    def validate_result(self, result, request):
        expected=sorted({f for order in frame_passes(len(request['frames']),request['prompt']['frame'],request['direction']) for f in order})
        if (result.get('schema')!=SCHEMA or result.get('status')!='completed' or result.get('job_id')!=request['job_id']
            or result.get('stage')!='sam3' or result.get('request_sha256')!=request['request_sha256']
            or result.get('source_sha256')!=request['source_sha256'] or result.get('evaluated_frames')!=expected):
            raise ValueError('Guided result identity or frame coverage differs from the request')
        component=ComponentVersion.model_validate(result['component'])
        if component.checkpoint_sha256!=request['checkpoint_sha256'] or component.version!='sam3@'+SAM3_COMMIT or component.config_sha256!=request['request_sha256']:
            raise ValueError('Guided model checkpoint or prompt provenance differs')
        track=VideoTrack.model_validate(result['track'])
        if track.track_id!=request['track_id'] or track.label_name!=request['label_name'] or track.label_id!=request['label_id']:raise ValueError('Guided result changed the requested object identity')
        if [o.frame_index for o in track.observations]!=expected:raise ValueError('Guided result omitted or duplicated frames')
        if any(o.bbox.xbr>request['width'] or o.bbox.ybr>request['height'] for o in track.observations):raise ValueError('Guided result extends outside the image')
        if request['prompt']['kind']=='box':
            anchor=next(o for o in track.observations if o.frame_index==request['prompt']['frame'])
            if anchor.bbox.model_dump()!=request['prompt']['bbox'] or anchor.outside:raise ValueError('Guided result changed the human anchor box')
        return track

    def apply(self, run_id, revision):
        self.check(revision)
        run=self.runs.get(run_id)
        if run is None:raise HTTPException(404,'Unknown tracking run')
        if run['status']!='ready':raise HTTPException(409,'Only a completed preview can be applied')
        if run['generation']!=self.generation or run['base_revision']!=self.revision:raise HTTPException(409,'Annotations changed since tracking started. Run again from the current revision.')
        self.snapshot()
        track=VideoTrack.model_validate(run['result']['track'])
        old=next((t for t in self.session.tracks.tracks if t.track_id==run['replace_track_id']),None)
        if old and run['direction']!='both':
            observed={o.frame_index for o in track.observations}
            track.observations=sorted(track.observations+[o for o in old.observations if o.frame_index not in observed],key=lambda o:o.frame_index)
        self.session.tracks.tracks=[t for t in self.session.tracks.tracks if t.track_id!=run['replace_track_id']]+[track]
        self.session.tracks.component_versions.append(ComponentVersion.model_validate(run['result']['component']))
        self.session.states[track.track_id]='unseen'
        run['status']='applied'
        response=self.changed()
        atomic_json(self.state_root/'runs'/run_id/'status.json',self.public(run))
        return response


def attach_workspace(app, session, root, **options):
    workspace=GuidedWorkspace(session,root,**options)
    session.workspace=workspace
    if session.job:workspace.restore();session.evaluate(session.threshold)
    app.state.workspace=workspace

    @app.get('/workspace/capabilities')
    def capabilities():
        return {'native_canvas':(Path(root)/'ui/dist/workspace.js').is_file(),'guided_tracking':workspace.capabilities(),'persistent_edits':True}

    @app.delete('/workspace/tracks/{track_id}')
    def delete(track_id:str,payload:Revision):
        with session.lock:return workspace.delete(track_id,payload.expected_revision)

    @app.post('/workspace/undo')
    def undo(payload:Revision):
        with session.lock:return workspace.undo(payload.expected_revision)

    @app.post('/workspace/repairs')
    def submit(payload:RepairRequest):
        with session.lock:return workspace.submit(payload)

    @app.get('/workspace/repairs')
    def runs():
        with session.lock:return [workspace.public(r) for r in sorted(workspace.runs.values(),key=lambda r:r.get('created_at',0)) if r['video_id']==(session.clip or {}).get('id')]

    @app.get('/workspace/repairs/{run_id}')
    def run(run_id:str):
        with session.lock:
            if run_id not in workspace.runs:raise HTTPException(404,'Unknown tracking run')
            return workspace.public(workspace.runs[run_id])

    @app.post('/workspace/repairs/{run_id}/resume')
    def resume(run_id:str,payload:Revision):
        with session.lock:return workspace.resume(run_id,payload.expected_revision)

    @app.get('/workspace/repairs/{run_id}/preview')
    def preview(run_id:str):
        with session.lock:
            record=workspace.runs.get(run_id)
            if record is None or record['status'] not in {'ready','applied'}:raise HTTPException(409,'Preview is not ready')
            return record['result']['track']

    @app.post('/workspace/repairs/{run_id}/apply')
    def apply(run_id:str,payload:Revision):
        with session.lock:return workspace.apply(run_id,payload.expected_revision)

    @app.get('/workspace',include_in_schema=False)
    def page():
        path=Path(root)/'ui/dist/index.html'
        return FileResponse(path) if path.is_file() else RedirectResponse('/mock/tasks/1/jobs/1')

    @app.get('/workspace-assets/{name}',include_in_schema=False)
    def asset(name:str):
        if name not in {'workspace.js','workspace.css','workspace.js.LEGAL.txt','cvat-source.json'}:raise HTTPException(404)
        path=Path(root)/'ui/dist'/name
        if not path.is_file():raise HTTPException(404)
        return FileResponse(path)

    return workspace
