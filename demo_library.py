"""Validated local video library; playback never invokes a cloud service."""
import hashlib
import json
from pathlib import Path
import re
from fastapi import HTTPException
from src.models.video import FrameManifest, TrackSet


class DemoLibrary:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def catalog(self):
        path = self.root / 'catalog.json'
        if not path.is_file():
            return {'videos': [], 'default_video_id': None}
        result = json.loads(path.read_text(encoding='utf-8'))
        if result.get('schema') != 'car-cam-demo-catalog/v1' or not isinstance(result.get('videos'), list):
            raise ValueError('Invalid processed video catalog')
        return result

    def entry(self, video_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', video_id):
            raise HTTPException(404, 'Unknown demo video')
        for row in self.catalog()['videos']:
            if row['id'] == video_id and row.get('directory') == video_id:
                return row
        raise HTTPException(404, 'Unknown demo video')

    def file(self, video_id, relative):
        self.entry(video_id)
        directory = (self.root / video_id).resolve()
        path = (directory / relative).resolve()
        if not directory.is_relative_to(self.root) or not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(404, 'Demo media is unavailable')
        return path

    def load(self, video_id):
        entry = self.entry(video_id)
        manifest = FrameManifest.model_validate_json(self.file(video_id, 'manifest.json').read_bytes())
        track_bytes = self.file(video_id, 'tracks.json').read_bytes()
        tracks = TrackSet.model_validate_json(track_bytes)
        if tracks.job_id != manifest.job_id or manifest.frame_count != entry['frame_count']:
            raise ValueError('Processed demo timeline does not match its tracks')
        if entry['source_sha256'] != manifest.source_sha256:
            raise ValueError('Processed demo source identity differs')
        movie = self.file(video_id, 'annotated.mp4')
        with movie.open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != entry['result_sha256']:
                raise ValueError('Processed video checksum differs')
        if not tracks.component_versions or tracks.component_versions[0].checkpoint_sha256 != entry['checkpoint_sha256']:
            raise ValueError('Processed model checkpoint identity differs')
        return entry, manifest, tracks
