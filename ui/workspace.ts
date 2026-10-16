import { Canvas, RectDrawingMethod } from './vendor/cvat-canvas/src/typescript/canvas';
import './workspace.css';

const $ = (id: string): any => document.getElementById(id);
let state: any, capabilities: any, canvas: any, currentFrame = Number(new URL(location.href).searchParams.get('frame')) || 0;
let selected: string | null = null, prompt: any = null, preview: any = null, activeRun: any = null;
let playTimer: any = null, pointLabel = 1, activeTool = 'select', busy = false;
const colors=['#36cfc9','#ffa940','#597ef7','#73d13d','#eb2f96','#ffc53d'];
const clientIDs=new Map<string, number>();
const images=new Map<string,Promise<ImageBitmap>>();
let idCounter=1;
function idFor(id: string){if(!clientIDs.has(id))clientIDs.set(id,idCounter++);return clientIDs.get(id);}
function message(text: string, error=false){$('message').textContent=text;$('message').classList.toggle('error',error);}
async function api(url: string, body?: any, method='POST'){
    const response=await fetch(url,body===undefined?{}:{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const result=await response.json();if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));return result;
}
async function action(fn:()=>Promise<any>){try{await fn();}catch(error:any){message(error.message,true);}}
function selectedTrack(){return state?.tracks?.tracks.find((t:any)=>t.track_id===selected);}
function observation(t:any){return t?.observations.find((o:any)=>o.frame_index===currentFrame&&!o.outside);}
function shape(id:string,box:any,color:string,label:string,lock=false){
    return {clientID:idFor(id),trackID:id,objectType:'track',shapeType:'rectangle',frame:currentFrame,
        points:[box.xtl,box.ytl,box.xbr,box.ybr],rotation:0,outside:false,occluded:false,hidden:false,lock,
        attributes:{},descriptions:[],elements:[],zOrder:lock?100:0,pinned:false,updated:state.revision,
        color,group:{id:0,color},label:{id:1,name:label,color,attributes:[],type:'rectangle'}};
}
function canvasStates(){
    const tracks=(state?.tracks?.tracks||[]).filter((t:any)=>t.review_status!=='rejected');
    const states=tracks.flatMap((t:any,i:number)=>{const o=observation(t);return o?[shape(t.track_id,o.bbox,colors[i%colors.length],t.label_name)]:[];});
    if(prompt?.frame===currentFrame){
        if(prompt.kind==='box')states.push(shape('prompt',prompt.bbox,'#ffffff','PROMPT',true));
        else prompt.points.forEach((p:any,i:number)=>{
            const s:any=shape('prompt:'+i,{xtl:p.x,ytl:p.y,xbr:p.x+1,ybr:p.y+1},p.label?'#52ff7d':'#ff4d4f',p.label?'Positive':'Negative',true);
            s.shapeType='points';s.points=[p.x,p.y];states.push(s);
        });
    }
    const o=observation(preview);if(o)states.push(shape('preview',o.bbox,'#fadb14','SAM3 PREVIEW',true));
    return states;
}
function image(frame:number){
    const url=`/mock/media/${encodeURIComponent(state.active_video.id)}/frames/${frame}`;
    if(!images.has(url))images.set(url,fetch(url).then(r=>{if(!r.ok)throw new Error('Frame image is unavailable');return r.blob();}).then(b=>createImageBitmap(b)));
    return images.get(url)!;
}
function setup(){
    if(!canvas||!state?.active_video)return;
    const states=canvasStates();
    // Release SVG resize/rotation controls while their previous shape still exists.
    canvas.activate(null);
    canvas.setSelectedObjects([]);
    canvas.setup({number:currentFrame,deleted:false,width:state.width,height:state.height,
        data:()=>image(currentFrame).then(bitmap=>({renderWidth:state.width,renderHeight:state.height,imageData:bitmap}))},states);
    const client=states.find(s=>s.trackID===selected&&!s.lock)?.clientID;
    canvas.activate(client??null);
    $('frame').value=currentFrame;$('seek').value=currentFrame;
    $('timelineInfo').textContent=`${(currentFrame/state.fps).toFixed(1)} s / ${(state.frame_count/state.fps).toFixed(1)} s`;
}
function renderPrompt(){
    $('promptSummary').textContent=prompt?`${prompt.kind==='box'?'Box':`${prompt.points.length} point(s)`} on frame ${prompt.frame}. ${selected&&$('replace').checked?'Will replace selected track.':'Will create a new track.'}`:'No prompt yet. Draw a box on the image.';
    $('runTracking').disabled=!prompt||busy||!capabilities?.guided_tracking.configured;
    $('removeTrack').disabled=!selected||busy;$('useBox').disabled=!observation(selectedTrack())||busy;
}
function select(id:string|null){selected=id;const t=selectedTrack();if(t)$('label').value=t.label_name;renderList();renderPrompt();setup();}
function renderList(){
    const tracks=state?.tracks?.tracks||[];$('trackCount').textContent=tracks.length;
    $('trackList').replaceChildren(...tracks.map((t:any,i:number)=>{
        const b=document.createElement('button');b.className='object'+(t.track_id===selected?' selected':'');
        b.style.borderLeftColor=colors[i%colors.length];b.dataset.track=t.track_id;
        const title=document.createElement('strong');title.textContent=t.label_name;
        const meta=document.createElement('span');meta.textContent=`${t.track_id.slice(-12)} · ${t.observations.filter((o:any)=>!o.outside).length} visible frames`;
        b.append(title,meta);b.onclick=()=>select(t.track_id);return b;
    }));
    if(!tracks.length)$('trackList').textContent='No tracks. Draw a prompt to add an object.';
}
function render(){
    const previous=$('video').value;
    $('video').replaceChildren(...state.videos.map((v:any)=>{const o=document.createElement('option');o.value=v.id;o.textContent=v.name;return o;}));
    $('video').value=state.active_video?.id||previous;
    if(selected&&!selectedTrack())selected=null;
    const label=$('label').value;$('label').replaceChildren(...state.labels.map((name:string)=>{const o=document.createElement('option');o.value=name;o.textContent=name;return o;}));
    if(state.labels.includes(label))$('label').value=label;
    currentFrame=Math.max(0,Math.min(state.frame_count-1,currentFrame));
    $('frame').max=$('seek').max=state.frame_count-1;$('frameTotal').textContent=`/ ${state.frame_count-1}`;
    $('savedVideo').href=`/mock/media/${state.active_video?.id}/video`;
    $('undo').disabled=!state.can_undo||busy;
    $('provider').textContent=capabilities.guided_tracking.message;
    const latest=state.history.at(-1);
    $('metrics').textContent=latest?`${latest.track_count??state.tracks.tracks.length} tracks · ${latest.flagged_tracks??0} flagged · ${state.frame_count} frames`:'';
    $('history').textContent=`Revision ${state.revision} · ${latest?.resolved_tracks??0} resolved`;
    renderList();renderPrompt();setup();
}
async function reload(){state=await api('/mock/state');render();}
function stop(){clearInterval(playTimer);playTimer=null;$('play').textContent='Play';}
function seek(frame:number){
    canvas.cancel();currentFrame=Math.max(0,Math.min(state.frame_count-1,Math.round(Number(frame)||0)));setup();renderPrompt();
    const u=new URL(location.href);u.searchParams.set('frame',String(currentFrame));history.replaceState(null,'',u);
}
function tool(value:string){
    stop();canvas.cancel();activeTool=value;
    if(value==='box')canvas.draw({enabled:true,shapeType:'rectangle',rectDrawingMethod:RectDrawingMethod.CLASSIC,crosshair:true});
    else if(value==='point')canvas.draw({enabled:true,shapeType:'points',numberOfPoints:1,crosshair:true});
    message(value==='select'?'Select a track, or draw a new prompt.':value==='box'?'Draw a box with two corners around one object.':`Click ${pointLabel?'inside the object':'on background to exclude it'}.`);
}
function useSelected(){
    const o=observation(selectedTrack());if(!o)throw new Error('This track is not visible on the current frame.');
    prompt={kind:'box',frame:currentFrame,bbox:{...o.bbox}};tool('select');renderPrompt();setup();
}
async function poll(runID:string){
    const r=await api('/workspace/repairs/'+runID);activeRun=r;
    $('resume').hidden=r.status!=='interrupted';
    $('runStatus').textContent=r.status==='running'||r.status==='queued'?`SAM3 ${r.status} · tracking one object from frame ${r.prompt.frame}…`:
        r.status==='ready'?`Preview ready · ${r.visible_frames} visible frames. Yellow boxes are the new track.`:
        r.status==='failed'?`Tracking failed: ${r.error}`:r.status==='applied'?'Tracking applied and saved.':r.status==='interrupted'?'Run interrupted. Resume uses the existing input and collects any submitted GPU result.':'';
    busy=['queued','running'].includes(r.status);renderPrompt();
    if(busy){setTimeout(()=>action(()=>poll(runID)),1500);return;}
    if(r.status==='ready'){
        preview=await api(`/workspace/repairs/${runID}/preview`);$('apply').hidden=false;$('discardPreview').hidden=false;setup();
    }
}
function initializeCanvas(){
    canvas=new Canvas();$('canvasHost').appendChild(canvas.html());
    const element=canvas.html();
    let firstFit=true;
    element.addEventListener('canvas.setup',()=>{$('canvasHost').dataset.ready='true';if(firstFit){firstFit=false;requestAnimationFrame(()=>canvas.fit());}});
    element.addEventListener('canvas.error',(e:any)=>{message(e.detail.exception.message,true);$('canvasFailure').hidden=false;});
    element.addEventListener('canvas.clicked',(e:any)=>{if(e.detail.state.trackID&&!e.detail.state.lock)select(e.detail.state.trackID);});
    element.addEventListener('canvas.find',(e:any)=>{
        if(activeTool!=='select')return;
        const {x,y,states,mouseEvent}=e.detail;
        const candidates=states.filter((s:any)=>!s.lock&&s.shapeType==='rectangle'&&x>=s.points[0]&&y>=s.points[1]&&x<=s.points[2]&&y<=s.points[3]);
        candidates.sort((a:any,b:any)=>(a.points[2]-a.points[0])*(a.points[3]-a.points[1])-(b.points[2]-b.points[0])*(b.points[3]-b.points[1]));
        const target=candidates[0];try{canvas.activate(target?.clientID||null);}catch{}
        if(mouseEvent?.type==='click'&&target)select(target.trackID);
    });
    element.addEventListener('canvas.drawn',(e:any)=>{
        const s=e.detail.state;
        if(s.shapeType==='rectangle'){
            const [x1,y1,x2,y2]=s.points;
            const b={xtl:Math.max(0,Math.min(x1,x2)),ytl:Math.max(0,Math.min(y1,y2)),xbr:Math.min(state.width,Math.max(x1,x2)),ybr:Math.min(state.height,Math.max(y1,y2))};
            if(b.xbr<=b.xtl||b.ybr<=b.ytl){message('Draw a non-empty box inside the image.',true);return;}
            prompt={kind:'box',frame:currentFrame,bbox:b};activeTool='select';
        }else if(s.shapeType==='points'){
            const [x,y]=s.points;
            if(x<0||y<0||x>=state.width||y>=state.height)return;
            if(prompt?.kind!=='points'||prompt.frame!==currentFrame)prompt={kind:'points',frame:currentFrame,points:[]};
            if(prompt.points.length>=16){message('Use at most 16 prompt points.',true);return;}
            prompt.points.push({x,y,label:pointLabel});activeTool='select';
        }
        renderPrompt();setup();message('Prompt captured. Run SAM3 to track this object across the video.');
    });
    element.addEventListener('canvas.edited',(e:any)=>action(async()=>{
        const {state:s,points:p}=e.detail;
        if(s.lock||!s.trackID)return;
        await api(`/mock/tracks/${encodeURIComponent(s.trackID)}/box`,{frame:currentFrame,bbox:{xtl:p[0],ytl:p[1],xbr:p[2],ybr:p[3]}});
        await reload();message('Box saved at this frame. Use it as a prompt to re-track the whole object.');
    }));
    new ResizeObserver(()=>canvas.fitCanvas()).observe($('canvasHost'));
    canvas.fitCanvas();
}
$('previous').onclick=()=>seek(currentFrame-1);$('next').onclick=()=>seek(currentFrame+1);
$('frame').onchange=()=>seek($('frame').value);$('seek').oninput=()=>seek($('seek').value);
$('fit').onclick=()=>canvas.fit();
$('selectTool').onclick=()=>tool('select');$('drawBox').onclick=()=>tool('box');
$('positive').onclick=()=>{pointLabel=1;tool('point');};$('negative').onclick=()=>{pointLabel=0;tool('point');};
$('clearPrompt').onclick=()=>{prompt=null;selected=null;tool('select');renderList();renderPrompt();setup();};
$('replace').onchange=renderPrompt;
$('useBox').onclick=()=>action(async()=>useSelected());
$('removeTrack').onclick=()=>action(async()=>{
    if(!selected)return;stop();canvas.cancel();
    await api('/workspace/tracks/'+encodeURIComponent(selected),{expected_revision:state.revision},'DELETE');
    selected=null;prompt=null;preview=null;$('apply').hidden=true;$('discardPreview').hidden=true;await reload();message('Object track removed from every frame. Undo is available.');
});
$('undo').onclick=()=>action(async()=>{await api('/workspace/undo',{expected_revision:state.revision});preview=null;$('apply').hidden=true;await reload();message('Track change undone and saved.');});
$('save').onclick=()=>message('Your track changes are saved locally. Export annotations to save a CVAT JSON copy.');
$('evaluate').onclick=()=>action(async()=>{await api('/mock/reevaluate',{iou_threshold:0.5});await reload();message('Review signals updated.');});
$('video').onchange=()=>action(async()=>{stop();canvas.cancel();await api(`/mock/videos/${encodeURIComponent($('video').value)}/select`,{});currentFrame=0;selected=null;prompt=null;preview=null;$('apply').hidden=true;await reload();});
$('play').onclick=()=>{if(playTimer){stop();return;}canvas.cancel();$('play').textContent='Pause';playTimer=setInterval(()=>{if(currentFrame>=state.frame_count-1){stop();return;}seek(currentFrame+1);},1000/state.fps);};
$('runTracking').onclick=()=>action(async()=>{
    stop();canvas.cancel();busy=true;renderPrompt();preview=null;$('apply').hidden=true;
    try{
        activeRun=await api('/workspace/repairs',{expected_revision:state.revision,prompt,label_name:$('label').value,
            replace_track_id:selected&&$('replace').checked?selected:null,direction:$('direction').value});
        await poll(activeRun.id);
    }catch(error){busy=false;renderPrompt();throw error;}
});
$('resume').onclick=()=>action(async()=>{activeRun=await api(`/workspace/repairs/${activeRun.id}/resume`,{expected_revision:state.revision});$('resume').hidden=true;await poll(activeRun.id);});
$('apply').onclick=()=>action(async()=>{
    await api(`/workspace/repairs/${activeRun.id}/apply`,{expected_revision:state.revision});
    selected=preview.track_id;preview=null;prompt=null;$('apply').hidden=true;$('discardPreview').hidden=true;
    await reload();message('SAM3 track applied across the requested frames and saved. Review signals updated.');$('runStatus').textContent='Applied.';
});
$('discardPreview').onclick=()=>{preview=null;$('apply').hidden=true;$('discardPreview').hidden=true;setup();};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes((e.target as HTMLElement).tagName))return;
    if(e.key==='Escape')tool('select');else if(e.key.toLowerCase()==='b')tool('box');else if(e.key==='ArrowRight')seek(currentFrame+1);else if(e.key==='ArrowLeft')seek(currentFrame-1);
});
(async()=>{
    capabilities=await api('/workspace/capabilities');state=await api('/mock/state');
    if(!state.active_video)throw new Error('Select a processed video in the fallback mock first.');
    initializeCanvas();render();message('Select an object to remove it, or draw a prompt to guide SAM3.');
    const runs=await api('/workspace/repairs');const last=runs.at(-1);if(last&&['running','queued','ready','interrupted'].includes(last.status))await poll(last.id);
})().catch(error=>{message(error.message,true);$('canvasFailure').hidden=false;});
