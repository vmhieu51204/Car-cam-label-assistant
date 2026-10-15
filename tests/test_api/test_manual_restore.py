import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.api.video import latest_manual_claude


def test_restore_reads_saved_result_and_filters_source_frame(tmp_path):
    directory = tmp_path / "manual-claude"
    directory.mkdir()
    saved = {"source_frame": 846, "proposals": [{"status": "accepted"}]}
    (directory / "saved.json").write_text(json.dumps(saved))
    service = SimpleNamespace(
        store=SimpleNamespace(get_video_job=lambda _: None), workspace=SimpleNamespace(job_dir=lambda _: tmp_path)
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(video_jobs=service)))
    assert latest_manual_claude(request, "job", 846) == {"result": saved}
    assert latest_manual_claude(request, "job", None) == {"result": saved}
    assert latest_manual_claude(request, "job", 847) == {"result": None}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for panel regression")
def test_panel_reload_preserves_current_frame_and_suggestion_frame():
    panel = Path(__file__).resolve().parents[2] / "integrations/cvat-autolabel/files/cvat-ui/autolabel-panel.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const helpers=source.slice(source.indexOf('    function currentSourceFrame()'),source.indexOf('    let sequenceClasses'));
let reloads=0; const storage=new Map();
const context={URL, panel:{},cvatJobId:17,manualResult:{source_frame:846},manualFrame:0,
 document:{querySelector:()=>({value:'120'})},
 location:{href:'http://localhost:8080/tasks/19/jobs/17?frame=0',reload:()=>reloads++},
 sessionStorage:{setItem:(k,v)=>storage.set(k,v)},
 window:{addEventListener:()=>{}},history:{state:{},replaceState:(s,t,url)=>context.location.href=String(url)}};
vm.createContext(context); vm.runInContext(helpers,context); context.reloadKeepingFrame();
assert.equal(context.currentSourceFrame(),120);
assert(!source.includes('manualFrame = saved.result.source_frame'));
assert(source.includes('manualFrame = currentSourceFrame();'));
assert.equal(new URL(context.location.href).searchParams.get('frame'),'120');
assert.equal(storage.get('autolabel-manual-frame-17'),'846');
assert.equal(storage.get('autolabel-reopen-17'),'1'); assert.equal(reloads,1);
"""
    subprocess.run(["node", "-e", script, str(panel)], check=True)
