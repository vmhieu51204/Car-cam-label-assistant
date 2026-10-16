import hashlib
import json
from pathlib import Path
from uuid import uuid4
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from demo_library import DemoLibrary
from local_mock import create_mock_app
from src.models.video import BoundingBox2D, ComponentVersion, FrameManifest, FrameRecord, TrackObservation, TrackSet, VideoTrack


@pytest.fixture
def library_root(tmp_path):
    directory=tmp_path/'test-video';directory.mkdir()
    job_id=str(uuid4())
    manifest=FrameManifest(job_id=job_id,source_name='test.mp4',source_sha256='a'*64,
        source_frame_count=35,sampling_strategy='all_frames',source_fps=10,target_fps=10,codec='fixture',
        width=80,height=40,duration_s=3.5,average_fps=10,variable_frame_rate=False,ffmpeg_version='test',
        frames=[FrameRecord(frame_index=f,source_frame_index=f,timestamp_s=f/10,source_timestamp_s=f/10,
            relative_path=f'frames/frame_{f:06d}.jpg',sha256='b'*64,width=80,height=40) for f in range(35)])
    encoded=manifest.model_dump_json().encode()
    (directory/'manifest.json').write_bytes(encoded)
    component=ComponentVersion(component='sam3-video-text',version='sam3@fixture',checkpoint_sha256='c'*64)
    tracks=TrackSet(job_id=job_id,manifest_sha256=hashlib.sha256(encoded).hexdigest(),prompt_sha256='d'*64,
        component_versions=[component],tracks=[VideoTrack(track_id='auto:000001',label_id=1,label_name='car',source='auto',
            observations=[TrackObservation(frame_index=34,bbox=BoundingBox2D(xtl=1,ytl=2,xbr=15,ybr=25))])])
    (directory/'tracks.json').write_text(tracks.model_dump_json(),encoding='utf-8')
    video=b'unit-test-media-bytes'*20
    (directory/'annotated.mp4').write_bytes(video)
    (directory/'poster.jpg').write_bytes(b'poster')
    (directory/'frames').mkdir();(directory/'frames/frame_000034.jpg').write_bytes(b'frame34')
    (tmp_path/'catalog.json').write_text(json.dumps({'schema':'car-cam-demo-catalog/v1','default_video_id':'test-video',
        'videos':[{'id':'test-video','directory':'test-video','name':'test.mp4','frame_count':35,'fps':10,'width':80,'height':40,
            'track_count':1,'source_sha256':'a'*64,'result_sha256':hashlib.sha256(video).hexdigest(),
            'checkpoint_sha256':'c'*64,'model_version':'sam3@fixture'}]}),encoding='utf-8')
    return tmp_path


def test_default_video_loaded_without_inference_or_reference_metrics(library_root):
    with TestClient(create_mock_app(library_root)) as client:
        state=client.get('/mock/state').json()
        assert state['active_video']['id']=='test-video'
        assert state['frame_count']==35 and state['fps']==10
        assert state['ground_truth'] is None
        assert state['history'][0]['reference_available'] is False
        assert state['history'][0]['precision'] is None
        assert client.get('/healthz').json()['weights_loaded'] is False
        assert client.get('/healthz').json()['mode']=='sam3-replay'


def test_video_range_requests_and_frame_boundaries(library_root):
    with TestClient(create_mock_app(library_root)) as client:
        response=client.get('/mock/media/test-video/video',headers={'Range':'bytes=0-9'})
        assert response.status_code==206 and len(response.content)==10
        assert response.headers['content-type']=='video/mp4'
        assert client.get('/mock/media/test-video/frames/34').content==b'frame34'
        assert client.get('/mock/media/test-video/frames/35').status_code==404
        assert client.get('/mock/media/test-video/frames/-1').status_code==404


def test_real_video_edits_support_late_frames_and_preserve_saved_results(library_root):
    with TestClient(create_mock_app(library_root)) as client:
        body={'frame':34,'bbox':{'xtl':4,'ytl':5,'xbr':20,'ybr':30}}
        assert client.post('/mock/tracks/auto:000001/box',json=body).status_code==200
        body['frame']=35
        assert client.post('/mock/tracks/auto:000001/box',json=body).status_code==422
        assert client.post('/mock/tracks/auto:000001/correct').status_code==409
        assert client.get('/mock/state').json()['tracks']['tracks'][0]['observations'][0]['bbox']['xtl']==4
    with TestClient(create_mock_app(library_root)) as client:
        assert client.get('/mock/state').json()['tracks']['tracks'][0]['observations'][0]['bbox']['xtl']==4
        assert json.loads((library_root/json.loads((library_root/'catalog.json').read_text())['videos'][0]['id']/'tracks.json').read_text())['tracks'][0]['observations'][0]['bbox']['xtl']==1


def test_switch_between_synthetic_and_saved_video(library_root):
    with TestClient(create_mock_app(library_root)) as client:
        assert client.post('/mock/synthetic').status_code==200
        assert client.get('/mock/state').json()['active_video'] is None
        assert client.post('/mock/videos/test-video/select').status_code==200
        assert client.get('/mock/state').json()['active_video']['id']=='test-video'
        assert client.post('/mock/videos/missing/select').status_code==404


def test_checksum_and_traversal_protection(library_root):
    library=DemoLibrary(library_root)
    with pytest.raises(HTTPException):library.file('test-video','../catalog.json')
    with pytest.raises(HTTPException):library.entry('../test-video')
    (library_root/'test-video/annotated.mp4').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):library.load('test-video')


def test_empty_library_retains_synthetic_workflow(tmp_path):
    with TestClient(create_mock_app(tmp_path)) as client:
        assert client.get('/mock/state').json()['job'] is None
        assert client.post('/mock/run').status_code==200
        assert client.get('/mock/state').json()['data_source']=='synthetic-mock'



def repair_payload(revision=0, **overrides):
    return {'expected_revision':revision,'prompt':{'kind':'box','frame':12,
        'bbox':{'xtl':3,'ytl':4,'xbr':18,'ybr':25}},'label_name':'car','replace_track_id':'auto:000001',
        'direction':'both',**overrides}


def guided_fixture_result(request, directory):
    from inference.sam3_guided import SCHEMA,SAM3_COMMIT,frame_passes
    frames=sorted({f for order in frame_passes(len(request['frames']),request['prompt']['frame'],request['direction']) for f in order})
    return {'schema':SCHEMA,'stage':'sam3','status':'completed','job_id':request['job_id'],
        'request_sha256':request['request_sha256'],'source_sha256':request['source_sha256'],
        'evaluated_frames':frames,'component':{'component':'sam3-guided','version':'sam3@'+SAM3_COMMIT,
            'checkpoint_sha256':request['checkpoint_sha256'],'config_sha256':request['request_sha256']},
        'track':{'track_id':request['track_id'],'label_id':request['label_id'],'label_name':request['label_name'],
            'source':'human_seed','observations':[{'frame_index':f,'bbox':request['prompt']['bbox']} for f in frames]}}


def wait_for_repair(client, run_id):
    import time
    for _ in range(100):
        result=client.get('/workspace/repairs/'+run_id).json()
        if result['status'] not in {'queued','running'}:return result
        time.sleep(.01)
    raise AssertionError('Repair did not finish')


def test_delete_whole_track_export_persist_undo_and_stale_revision(library_root):
    with TestClient(create_mock_app(library_root)) as client:
        assert client.request('DELETE','/workspace/tracks/auto:000001',json={'expected_revision':0}).status_code==200
        assert client.get('/mock/state').json()['tracks']['tracks']==[]
        assert client.get('/mock/export').json()['tracks']==[]
        assert client.post('/workspace/undo',json={'expected_revision':0}).status_code==409
        assert client.post('/workspace/undo',json={'expected_revision':1}).status_code==200
        assert len(client.get('/mock/export').json()['tracks'])==1
        assert client.request('DELETE','/workspace/tracks/auto:000001',json={'expected_revision':2}).status_code==200
    with TestClient(create_mock_app(library_root)) as client:
        assert client.get('/mock/state').json()['tracks']['tracks']==[]
        assert client.get('/mock/state').json()['revision']==3


def test_guided_preview_is_not_applied_until_requested(library_root):
    with TestClient(create_mock_app(library_root,repair_runner=guided_fixture_result)) as client:
        run=client.post('/workspace/repairs',json=repair_payload()).json()
        assert wait_for_repair(client,run['id'])['status']=='ready'
        assert len(client.get('/mock/state').json()['tracks']['tracks'][0]['observations'])==1
        preview=client.get('/workspace/repairs/'+run['id']+'/preview').json()
        assert [o['frame_index'] for o in preview['observations']]==list(range(35))
        assert client.post('/workspace/repairs/'+run['id']+'/apply',json={'expected_revision':0}).status_code==200
        assert len(client.get('/mock/state').json()['tracks']['tracks'][0]['observations'])==35
        assert client.post('/workspace/repairs/'+run['id']+'/apply',json={'expected_revision':1}).status_code==409


def test_guided_preview_cannot_overwrite_edits_or_deleted_track(library_root):
    with TestClient(create_mock_app(library_root,repair_runner=guided_fixture_result)) as client:
        run=client.post('/workspace/repairs',json=repair_payload()).json()
        assert wait_for_repair(client,run['id'])['status']=='ready'
        client.request('DELETE','/workspace/tracks/auto:000001',json={'expected_revision':0})
        assert client.post('/workspace/repairs/'+run['id']+'/apply',json={'expected_revision':1}).status_code==409
        assert client.get('/mock/export').json()['tracks']==[]


def test_failed_model_provenance_keeps_original_tracks(library_root):
    def wrong_checkpoint(request,directory):
        result=guided_fixture_result(request,directory);result['component']['checkpoint_sha256']='f'*64
        return result
    with TestClient(create_mock_app(library_root,repair_runner=wrong_checkpoint)) as client:
        run=client.post('/workspace/repairs',json=repair_payload()).json()
        assert wait_for_repair(client,run['id'])['status']=='failed'
        assert len(client.get('/mock/state').json()['tracks']['tracks'][0]['observations'])==1
        assert client.get('/workspace/repairs/'+run['id']+'/preview').status_code==409


def test_prompt_bounds_positive_points_and_frame_validation(library_root):
    with TestClient(create_mock_app(library_root,repair_runner=guided_fixture_result)) as client:
        for prompt in [
            {'kind':'box','frame':35,'bbox':{'xtl':1,'ytl':1,'xbr':8,'ybr':8}},
            {'kind':'box','frame':1,'bbox':{'xtl':1,'ytl':1,'xbr':81,'ybr':8}},
            {'kind':'points','frame':1,'points':[{'x':1,'y':1,'label':0}]},
            {'kind':'points','frame':1,'points':[{'x':80,'y':1,'label':1}]},
        ]:
            response=client.post('/workspace/repairs',json=repair_payload(prompt=prompt))
            assert response.status_code==422,response.text


def test_incomplete_model_coverage_cannot_be_previewed(library_root):
    def missing_frame(request,directory):
        result=guided_fixture_result(request,directory);result['track']['observations'].pop()
        return result
    with TestClient(create_mock_app(library_root,repair_runner=missing_frame)) as client:
        run=client.post('/workspace/repairs',json=repair_payload()).json()
        assert wait_for_repair(client,run['id'])['status']=='failed'
        assert client.get('/mock/state').json()['revision']==0
