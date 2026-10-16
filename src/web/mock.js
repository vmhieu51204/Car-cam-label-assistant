(() => {
'use strict';
const $ = id => document.getElementById(id);
let state = null, timer = null, view = 'video', drawToken = 0;
const frameImages = new Map();
let drawingPrompt=false, dragStart=null, pendingBox=null, repair=null, repairPreview=null, trackingBusy=false, trackingConfigured=false;
const realVideo = () => Boolean(state?.active_video);
const frameCount = () => state?.frame_count || 24;
const fps = () => state?.fps || 12;
async function api(url, body) {
    const r = await fetch(url, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const result = await r.json();
    if (!r.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
    return result;
}
function selectedFrame() { return Math.max(0,Math.min(frameCount()-1,Number($('frame').value)||0)); }
function boxFor(track, frame) { return track?.observations.find(o => o.frame_index === frame && !o.outside); }
function updateBoxInput() {
    const obs = boxFor(state?.tracks?.tracks.find(t => t.track_id === $('track').value), selectedFrame());
    $('bbox').value = obs ? [obs.bbox.xtl,obs.bbox.ytl,obs.bbox.xbr,obs.bbox.ybr].join(',') : '';
}
function setView(value) {
    if(value === 'video' && timer){clearInterval(timer);timer=null;$('play').textContent='Play';}
    view = value;
    $('resultVideo').hidden = !realVideo() || view !== 'video';
    $('scene').hidden = realVideo() && view === 'video';
    if (view === 'frame') { $('resultVideo').pause(); draw(); }
}
function paintBoxes(c, frame) {
    const overlay=repairPreview?.observations.find(o=>o.frame_index===frame&&!o.outside);
    const guide=pendingBox?.frame===frame?pendingBox.bbox:overlay?.bbox;
    if(guide){c.strokeStyle="#fadb14";c.lineWidth=4;c.strokeRect(guide.xtl,guide.ytl,guide.xbr-guide.xtl,guide.ybr-guide.ytl);}
    for (const t of state?.tracks?.tracks || []) {
        const o=boxFor(t,frame);if(!o)continue;const b=o.bbox;
        c.strokeStyle='#61a7ff';c.lineWidth=3;c.strokeRect(b.xtl,b.ytl,b.xbr-b.xtl,b.ybr-b.ytl);
        const text=`${t.label_name} · ${t.track_id}`;c.font='14px system-ui';
        const top=Math.max(0,b.ytl-24);c.fillStyle='#124b8d';c.fillRect(b.xtl,top,c.measureText(text).width+12,24);c.fillStyle='white';c.fillText(text,b.xtl+5,top+17);
    }
}
function imageFor(frame) {
    const url=`/mock/media/${encodeURIComponent(state.active_video.id)}/frames/${frame}`;
    if (!frameImages.has(url)) {
        const image=new Image();
        const promise=new Promise((resolve,reject)=>{image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Could not load video frame.'));});
        image.src=url;frameImages.set(url,promise);
        if(frameImages.size>20)frameImages.delete(frameImages.keys().next().value);
    }
    return frameImages.get(url);
}
async function draw() {
    const token=++drawToken, c=$('scene').getContext('2d'), f=selectedFrame();
    if(realVideo()) {
        try {
            const image=await imageFor(f);
            if(token!==drawToken)return;
            c.drawImage(image,0,0,$('scene').width,$('scene').height);paintBoxes(c,f);
        } catch(error) {$('message').textContent=error.message;}
        return;
    }
    c.fillStyle='#a7c3d0';c.fillRect(0,0,960,540);c.fillStyle='#6e8589';c.fillRect(0,165,960,100);
    for(let i=0;i<10;i++){c.fillStyle=i%2?'#879da0':'#738b90';c.fillRect(i*110-15,80+(i%3)*30,86,185);}
    c.fillStyle='#bcbfb3';c.fillRect(0,250,960,28);c.fillStyle='#46515b';c.fillRect(0,278,960,262);
    c.strokeStyle='#c7c9bd';c.lineWidth=4;c.setLineDash([55,35]);c.beginPath();c.moveTo(0,440);c.lineTo(960,440);c.stroke();c.setLineDash([]);
    const truth=state?.ground_truth?.[f]||[['car',[100+f*12,270,255+f*12,370]],...(f>=8?[['person',[720-f*4,230,748-f*4,330]]]:[])];
    for(const [label,b] of truth){const [x,y,x2,y2]=b;c.fillStyle=label==='car'?'#c45245':'#f4bb61';
        if(label==='car'){c.fillRect(x,y+25,x2-x,y2-y-34);c.fillStyle='#bbdbeb';c.fillRect(x+30,y+4,85,36);c.fillStyle='#202833';for(const xx of [x+25,x2-25]){c.beginPath();c.arc(xx,y2-7,15,0,7);c.fill();}}
        else{c.beginPath();c.arc((x+x2)/2,y+9,9,0,7);c.fill();c.fillRect(x+6,y+22,x2-x-12,45);c.strokeStyle='#f4bb61';c.lineWidth=7;c.beginPath();c.moveTo(x+8,y+65);c.lineTo(x+6,y2);c.moveTo(x2-8,y+65);c.lineTo(x2-6,y2);c.stroke();}
        if($('showTruth').checked){c.strokeStyle='#68f0a1';c.lineWidth=2;c.setLineDash([6,4]);c.strokeRect(x,y,x2-x,y2-y);c.setLineDash([]);}
    }
    paintBoxes(c,f);c.fillStyle='rgba(0,0,0,.5)';c.fillRect(16,15,240,32);c.fillStyle='white';c.font='15px system-ui';c.fillText(`SYNTHETIC SAMPLE · frame ${f}`,26,37);
}
function metric(label,value) {
    const d=document.createElement('div');d.className='metric';const b=document.createElement('b');b.textContent=value;const s=document.createElement('small');s.textContent=label;d.append(b,s);$('metrics').appendChild(d);
}
function render() {
    const tracks=state?.tracks?.tracks||[],previous=$('track').value,isReal=realVideo(),clip=state?.active_video;
    $('track').replaceChildren(...tracks.map(t=>{const o=document.createElement('option');o.value=t.track_id;o.textContent=`${t.label_name} · ${t.track_id}`;return o;}));
    if(tracks.some(t=>t.track_id===previous))$('track').value=previous;
    $('videoSelect').replaceChildren(...(state?.videos||[]).map(v=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.name;return o;}));
    const synthetic=document.createElement('option');synthetic.value='synthetic';synthetic.textContent='Synthetic test sample';$('videoSelect').appendChild(synthetic);$('videoSelect').value=clip?.id||'synthetic';
    const promptLabel=$('promptLabel').value;$('promptLabel').replaceChildren(...(state?.labels||[]).map(name=>{const o=document.createElement('option');o.value=name;o.textContent=name;return o;}));if(state?.labels.includes(promptLabel))$('promptLabel').value=promptLabel;
    $('removeWhole').disabled=!tracks.length;$('undoWhole').disabled=!state?.can_undo;$('trackBox').disabled=!isReal||!trackingConfigured||trackingBusy;
    $('frame').max=$('seek').max=String(frameCount()-1);$('frame').value=selectedFrame();$('seek').value=selectedFrame();
    $('scene').width=state?.width||960;$('scene').height=state?.height||540;
    $('evaluate').disabled=!state?.job;$('edit').disabled=!state?.job;$('export').hidden=!state?.job;
    $('showTruth').parentElement.hidden=isReal;$('thresholdControls').hidden=isReal;$('viewControls').hidden=!isReal;
    $('modeBadge').textContent=isReal?'Saved SAM3 result · offline replay':'Local mock · synthetic predictions';
    $('run').textContent=isReal?'Reload saved result':'Run synthetic sample';
    $('introText').textContent=isReal?'Saved SAM3 detections and stitched tracks from your video. Open AutoLabel video to review issues, or edit a frame below.':'Exercise the original AutoLabel panel with synthetic predictions and known reference boxes.';
    $('mediaCaption').textContent=isReal?`${clip.name} · ${frameCount()} frames at ${fps()} FPS · Result video shows the saved SAM3 run. Box edits appear in the frame editor.`:'Green: synthetic reference · Blue: prediction · 24 frames at 12 FPS.';
    $('accuracyNote').textContent=isReal?'No reference annotations were supplied. Re-evaluation updates review signals; accuracy is unavailable.':'';
    $('evaluate').textContent=isReal?'Re-evaluate review':'Re-evaluate';
    $('referenceHelp').textContent=isReal?'Edit boxes at the current frame. All model tracks remain pending human review.':'“Use reference” applies a synthetic reference correction.';
    $('modelInfo').textContent=isReal?`${clip.model_version} · ${clip.track_count} tracks. Playback needs no GPU, API keys, or cloud connection.`:'No model weights are loaded for the synthetic fixture.';
    if(isReal && $('resultVideo').dataset.clip!==`${clip.id}:${clip.result_sha256}`){$('resultVideo').dataset.clip=`${clip.id}:${clip.result_sha256}`;$('resultVideo').src=`/mock/media/${encodeURIComponent(clip.id)}/video`;$('resultVideo').poster=`/mock/media/${encodeURIComponent(clip.id)}/poster`;$('downloadVideo').href=$('resultVideo').src;$('downloadVideo').download=`${clip.id}-sam3.mp4`;view='video';}
    const latest=state?.history.at(-1);$('metrics').replaceChildren();
    if(latest){
        if(isReal){metric('Tracks',latest.track_count);metric('Flagged',latest.flagged_tracks);metric('Frames',frameCount());}
        else for(const [label,value] of [['Precision',latest.precision],['Recall',latest.recall],['F1',latest.f1]])metric(label,value===null?'—':`${Math.round(value*100)}%`);
    }
    $('history').textContent=(state?.history||[]).slice(-4).map(r=>isReal?`#${r.iteration} · ${r.track_count} tracks · ${r.flagged_tracks} flagged · ${r.resolved_tracks} resolved`:`#${r.iteration} · IoU ${r.iou_threshold} · TP ${r.tp} FP ${r.fp} FN ${r.fn}`).join('\n');
    $('tracks').replaceChildren();
    for(const t of tracks){const d=document.createElement('div');d.className='track';const text=document.createElement('span');text.textContent=`${t.label_name} · ${t.track_id}`;const button=document.createElement('button');button.textContent=isReal?'View':'Use reference';
        button.onclick=()=>act(async()=>{if(isReal){$('track').value=t.track_id;const row=state.review?.tracks.find(r=>r.track_id===t.track_id);seek(row?.source_frame??t.observations[0].frame_index);return;}
            await api(`/mock/tracks/${encodeURIComponent(t.track_id)}/correct`,{});await reload();refreshPanel();$('message').textContent='Synthetic correction applied; evaluation updated.';});
        d.append(text,button);$('tracks').appendChild(d);
    }
    if(!tracks.length)$('tracks').textContent='Run a sample to create tracks.';
    updateBoxInput();setView(isReal?view:'frame');draw();
}
async function reload(){state=await api('/mock/state');render();}
function refreshPanel(){if(document.querySelector('.autolabel-dock'))document.querySelector('.autolabel-open')?.click();}
async function act(fn){try{await fn();}catch(e){$('message').textContent=e.message;}}
$('run').onclick=()=>act(async()=>{await api('/mock/run',{});await reload();$('message').textContent=realVideo()?'Saved model result loaded.':'Synthetic run complete.';document.querySelector('.autolabel-open')?.click();});
$('videoSelect').onchange=()=>act(async()=>{if(timer){clearInterval(timer);timer=null;}$('play').textContent='Play';$('resultVideo').pause();const id=$('videoSelect').value;if(id!=='synthetic')view='video';await api(id==='synthetic'?'/mock/synthetic':`/mock/videos/${encodeURIComponent(id)}/select`,{});$('frame').value=0;frameImages.clear();await reload();refreshPanel();});
$('evaluate').onclick=()=>act(async()=>{await api('/mock/reevaluate',{iou_threshold:Number($('threshold').value)});await reload();refreshPanel();$('message').textContent='Re-evaluation complete.';});
$('edit').onclick=()=>act(async()=>{const b=$('bbox').value.split(',').map(Number);if(b.length!==4||b.some(x=>!Number.isFinite(x)))throw new Error('Enter four box coordinates.');await api(`/mock/tracks/${encodeURIComponent($('track').value)}/box`,{frame:selectedFrame(),bbox:{xtl:b[0],ytl:b[1],xbr:b[2],ybr:b[3]}});await reload();refreshPanel();setView('frame');$('message').textContent='Box edited; evaluation updated.';});
function seek(value){$('frame').value=Math.max(0,Math.min(frameCount()-1,Number(value)||0));$('seek').value=$('frame').value;updateBoxInput();if(realVideo()){setView('frame');try{$('resultVideo').currentTime=selectedFrame()/fps();}catch(error){/* Media metadata may still be loading. */}}draw();const u=new URL(location.href);u.searchParams.set('frame',$('frame').value);history.replaceState(null,'',u);}
$('frame').addEventListener('change',()=>seek($('frame').value));$('frame').addEventListener('input',()=>seek($('frame').value));$('seek').oninput=()=>seek($('seek').value);$('track').onchange=updateBoxInput;$('showTruth').onchange=draw;
$('play').onclick=()=>{if(realVideo()&&view==='video'){if($('resultVideo').paused)$('resultVideo').play().catch(e=>{$('message').textContent=e.message;});else $('resultVideo').pause();return;}if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';}else{$('play').textContent='Pause';timer=setInterval(()=>seek((selectedFrame()+1)%frameCount()),1000/fps());}};
$('resultVideo').addEventListener('play',()=>{$('play').textContent='Pause';});$('resultVideo').addEventListener('pause',()=>{$('play').textContent='Play';});
$('resultVideo').addEventListener('timeupdate',()=>{if(view!=='video')return;const f=Math.min(frameCount()-1,Math.floor($('resultVideo').currentTime*fps()));$('frame').value=f;$('seek').value=f;});
$('resultView').onclick=()=>setView('video');$('frameView').onclick=()=>setView('frame');
$('save').onclick=()=>{$('message').textContent='Edits are held in this local server session. Export annotations to keep a copy.';};

$('removeWhole').onclick=()=>act(async()=>{
    await apiDelete('/workspace/tracks/'+encodeURIComponent($('track').value),{expected_revision:state.revision});
    pendingBox=null;repairPreview=null;await reload();refreshPanel();setView('frame');$('message').textContent='Object removed from every frame. Undo is available.';
});
async function apiDelete(url,body){const r=await fetch(url,{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const result=await r.json();if(!r.ok)throw new Error(result.detail);return result;}
$('undoWhole').onclick=()=>act(async()=>{await api('/workspace/undo',{expected_revision:state.revision});await reload();refreshPanel();setView('frame');});
function parsePromptBox(){const b=$('bbox').value.split(',').map(Number);if(b.length!==4||b.some(n=>!Number.isFinite(n)))throw new Error('Draw a box or enter four box coordinates.');return {xtl:b[0],ytl:b[1],xbr:b[2],ybr:b[3]};}
$('drawPrompt').onclick=()=>{setView('frame');drawingPrompt=true;$('scene').style.cursor='crosshair';$('message').textContent='Drag a box around one object.';};
function pointerPosition(e){const canvas=$('scene'),r=canvas.getBoundingClientRect(),scale=Math.min(r.width/canvas.width,r.height/canvas.height);return [Math.max(0,Math.min(canvas.width,(e.clientX-r.left-(r.width-canvas.width*scale)/2)/scale)),Math.max(0,Math.min(canvas.height,(e.clientY-r.top-(r.height-canvas.height*scale)/2)/scale))];}
$('scene').onpointerdown=e=>{if(!drawingPrompt)return;dragStart=pointerPosition(e);$('scene').setPointerCapture(e.pointerId);};
$('scene').onpointerup=e=>{if(!drawingPrompt||!dragStart)return;const end=pointerPosition(e);const b={xtl:Math.min(dragStart[0],end[0]),ytl:Math.min(dragStart[1],end[1]),xbr:Math.max(dragStart[0],end[0]),ybr:Math.max(dragStart[1],end[1])};dragStart=null;drawingPrompt=false;$('scene').style.cursor='';if(b.xbr<=b.xtl||b.ybr<=b.ytl)return;pendingBox={frame:selectedFrame(),bbox:b};$('bbox').value=[b.xtl,b.ytl,b.xbr,b.ybr].map(n=>Math.round(n)).join(',');draw();$('message').textContent='Box captured. Run SAM3 to track this object across all frames.';};
async function pollRepair(){
    repair=await api('/workspace/repairs/'+repair.id);trackingBusy=['queued','running'].includes(repair.status);$('trackBox').disabled=trackingBusy||!trackingConfigured;
    $('repairStatus').textContent=repair.status==='failed'?repair.error:'SAM3: '+repair.status;
    if(trackingBusy){setTimeout(()=>act(pollRepair),1500);return;}
    if(repair.status==='ready'){repairPreview=await api('/workspace/repairs/'+repair.id+'/preview');$('applyRepair').hidden=false;setView('frame');draw();$('repairStatus').textContent='Preview ready. Yellow boxes show the new track.';}
}
$('trackBox').onclick=()=>act(async()=>{
    const bbox=parsePromptBox();repair=await api('/workspace/repairs',{expected_revision:state.revision,prompt:{kind:'box',frame:selectedFrame(),bbox},label_name:$('promptLabel').value,replace_track_id:$('replaceWhole').checked?$('track').value:null,direction:'both'});await pollRepair();
});
$('applyRepair').onclick=()=>act(async()=>{await api('/workspace/repairs/'+repair.id+'/apply',{expected_revision:state.revision});pendingBox=null;repairPreview=null;$('applyRepair').hidden=true;await reload();refreshPanel();setView('frame');$('repairStatus').textContent='Track applied and saved.';});
api('/workspace/capabilities').then(c=>{trackingConfigured=c.guided_tracking.configured;if(state)render();}).catch(()=>{});

window.addEventListener('focus',()=>act(reload));
if(!localStorage.getItem('autolabel-language'))localStorage.setItem('autolabel-language','en');
$('frame').value=Number(new URL(location.href).searchParams.get('frame'))||0;act(reload);
})();
