/* Privacy-first VM workflow. Credentials stay on the server and never enter browser storage. */
(() => {
    'use strict';
    const api = '/autolabel/api/v1';
    const randomID=()=>Array.from(crypto.getRandomValues(new Uint8Array(16)),v=>v.toString(16).padStart(2,'0')).join('');
    const element = (tag, text = '', parent = null) => {
        const node = document.createElement(tag); node.textContent = text;
        if (parent) parent.appendChild(node); return node;
    };
    const csrf = () => decodeURIComponent(document.cookie.split('; ').find(v => v.startsWith('csrftoken='))?.slice(10) || '');
    async function request(path, method = 'GET', body = undefined) {
        const headers = { 'X-CSRFToken': csrf() };
        if (body && !(body instanceof FormData)) headers['Content-Type'] = 'application/json';
        const response = await fetch(api + path, { method, credentials: 'same-origin', headers,
            body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body) });
        const payload = await response.json();
        if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `HTTP ${response.status}`);
        return payload;
    }
    function bridge(action, extra = {}) {
        return new Promise((resolve, reject) => {
            const requestID = randomID();
            const done = (event) => {
                if (event.detail.requestID !== requestID) return;
                clearTimeout(timer); window.removeEventListener('autolabel:vm-reply', done);
                if (event.detail.ok) resolve(event.detail); else reject(new Error(event.detail.error));
            };
            const timer = setTimeout(() => { window.removeEventListener('autolabel:vm-reply', done); reject(new Error('Mở job CVAT trước thao tác này')); }, 6000);
            window.addEventListener('autolabel:vm-reply', done);
            window.dispatchEvent(new CustomEvent('autolabel:vm-command', { detail: { requestID, action, ...extra } }));
        });
    }
    const style = element('style', '', document.head);
    style.textContent = `
      .cvat-annotation-page.vm-docked{display:flex!important;flex-direction:column!important}
      .vm-docked>.cvat-annotation-layout-content{flex:1;min-height:0;height:auto!important}
      #vm-panel{box-sizing:border-box;background:#fff;color:#222;border-top:2px solid #315da8;flex:0 0 255px;height:255px;overflow:hidden;z-index:30;width:100%;padding:8px}
      #vm-panel.vm-floating{position:fixed;bottom:0;left:0;right:0;height:285px;z-index:1200}
      #vm-panel.vm-collapsed{height:48px;flex-basis:48px}
      #vm-panel nav{display:flex;gap:8px;align-items:center;min-height:32px}
      #vm-panel section{height:calc(100% - 42px);overflow:auto;display:flex;gap:10px;flex-wrap:wrap;align-content:start}
      #vm-panel label{display:inline-flex;gap:6px;align-items:center;margin:4px}
      #vm-panel input:not([type=checkbox]),#vm-panel select{max-width:190px}
      #vm-panel button,.vm-dialog button{margin:3px;padding:5px 9px;cursor:pointer}
      #vm-panel [role=alert]{color:#a31515;width:100%}
      .vm-progress{min-width:230px;flex:1}.vm-progress progress{width:100%}
      #vm-timeline{width:100%;height:18px;cursor:pointer;border:1px solid #888}
      .vm-results{display:flex;gap:10px}.vm-results img{max-width:320px;max-height:140px}
      .vm-dialog{max-width:90vw;max-height:85vh;width:600px;border:1px solid #888;padding:20px;overflow:auto;color:#222}
      .vm-dialog::backdrop{background:#0005}.vm-dialog label{display:block;margin:12px 0}
      .vm-dialog input{display:block;box-sizing:border-box;width:100%}.vm-dialog pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:180px;overflow:auto}
      .vm-review{width:100%;height:110px;overflow:auto}.vm-review table{width:100%}
      @media(max-height:800px){#vm-panel{flex-basis:220px;height:220px}}
    `;
    let job = null, panel = null, busy = false, route = '', pending = 0, shown = 0, reviewPage = 1;
    let selected = null, overlay = null, closed = false;
    const launcher = element('button', 'AutoLabel VM', document.body);
    launcher.style.cssText = 'position:fixed;right:20px;bottom:10px;z-index:1000;padding:8px';
    const nodes = {};
    function button(parent, text, callback) {
        const b = element('button', text, parent); b.type = 'button';
        b.onclick = async () => {
            if (busy) return;
            busy = true; b.disabled = true;
            try { await callback(); if(nodes.error) nodes.error.textContent = ''; }
            catch (error) { if(nodes.error) nodes.error.textContent = error.message; else throw error; }
            finally { busy = false; b.disabled = false; }
        }; return b;
    }
    function input(parent, name, type = 'text', value = '') {
        const label = element('label', name, parent); const n = element('input', '', label);
        n.type = type; n.value = value; return n;
    }
    function section(name) { const s = element('section', '', panel); s.dataset.tab = name; s.hidden = true; return s; }
    function tab(name) {
        panel.querySelectorAll('section').forEach(s => { s.hidden = s.dataset.tab !== name; s.style.display = s.hidden ? 'none' : 'flex'; });
        panel.querySelectorAll('[data-nav]').forEach(n => n.setAttribute('aria-selected', String(n.dataset.nav === name)));
        localStorage.setItem('autolabel-vm-tab', name);
        if (name === 'review' && job) loadReview().catch(e => { nodes.error.textContent = e.message; });
    }
    function mount() {
        if (panel && panel.isConnected) return;
        panel = element('aside', '', document.body); panel.id = 'vm-panel'; panel.setAttribute('aria-label', 'AutoLabel VM');
        const nav = element('nav', '', panel); nav.setAttribute('role', 'tablist');
        [['run','Chạy'],['review','Duyệt'],['three-d','3D'],['vlm','VLM']].forEach(([id,label]) => {
            const b = button(nav, label, () => tab(id)); b.dataset.nav = id; b.setAttribute('role','tab');
        });
        button(nav,'Settings VLM',settingsDialog);
        button(nav,'Thu/Mở',() => { panel.classList.toggle('vm-collapsed'); localStorage.setItem('autolabel-vm-collapsed',String(panel.classList.contains('vm-collapsed'))); });
        button(nav,'Đóng',() => { closed = true; panel.hidden = true; launcher.hidden = false; document.querySelector('.vm-docked')?.classList.remove('vm-docked'); removeOverlay(); });
        nodes.notice = element('span','',nav);
        nodes.error = element('p','',nav); nodes.error.setAttribute('role','alert');
        const run = section('run');
        const upload = element('div','',run);
        nodes.video = input(upload,'Video','file'); nodes.video.accept = 'video/*';
        nodes.classes = input(upload,'Class CVAT (phân cách dấu phẩy)','text','car,person');
        element('p','YOLO + ByteTrack + SAM3',upload);
        nodes.audit = input(upload,'Rà VLM tối đa 20 frame sau SAM','checkbox');
        button(upload,'Upload → che → chạy',async () => {
            if (!nodes.video.files[0]) throw new Error('Chọn video để tạo task mới');
            const data = new FormData(); data.append('video',nodes.video.files[0]);
            data.append('options',JSON.stringify({ classes: nodes.classes.value.split(',').map(v=>v.trim()), variant:'sam3',pipeline:'hybrid_frame',vlm_enabled:nodes.audit.checked }));
            job = await request('/vm/jobs','POST',data); localStorage.setItem('autolabel-vm-job',job.job_id); shown = 0;
            nodes.video.value = ''; await poll();
        });
        button(upload,'Retry',async () => { requireJob(); job = await request(`/vm/jobs/${job.job_id}/retry`,'POST'); });
        button(upload,'Dừng',async () => { requireJob(); job = await request(`/vm/jobs/${job.job_id}/cancel`,'POST'); });
        button(upload,'Thêm vùng che',privacyDialog);
        nodes.link = element('a','',upload);
        const progress = element('div','',run); progress.className = 'vm-progress';
        for (const [key,labelText] of [['privacy_frames','Che nhận dạng'],['delivered_frames','Tải ảnh'],['processed_frames','SAM có kết quả'],['complete_frames','Đủ mọi class'],['published_frames','Đã gửi CVAT']]) {
            const row = element('div','',progress); nodes[key+'Text'] = element('span',labelText,row); nodes[key] = element('progress','',row);
        }
        nodes.status = element('p','Chưa có job',progress);
        nodes.sync = button(progress,'Đồng bộ sau Save',async () => { await bridge('sync'); shown = pending; nodes.notice.textContent='Đã đồng bộ'; });
        nodes.timeline = element('canvas','',progress); nodes.timeline.id = 'vm-timeline'; nodes.timeline.width = 1000; nodes.timeline.height = 18;
        nodes.timeline.setAttribute('aria-label','Timeline: xám chưa xử lý, vàng một phần, xanh đủ class, đỏ lỗi');
        nodes.timeline.onclick = async (event) => {
            try { requireJob(); await bridge('jump',{frame: Math.min(job.total_frames-1,Math.floor(event.offsetX/nodes.timeline.clientWidth*job.total_frames))}); }
            catch (error) { nodes.error.textContent=error.message; }
        };
        element('small','Timeline: xám chưa xử lý · vàng một phần · xanh đủ class · đỏ lỗi. Nhãn luôn chờ người duyệt.',progress);
        const review = section('review');
        nodes.reviewClass = input(review,'Class'); nodes.reviewViewed = input(review,'unseen/seen/resolved');
        button(review,'Lọc',async()=> { reviewPage=1; await loadReview(); });
        button(review,'Trước',async()=> { reviewPage=Math.max(1,reviewPage-1); await loadReview(); });
        button(review,'Sau',async()=> { reviewPage++; await loadReview(); });
        nodes.reviewLabel = element('span','',review); nodes.reviewRows = element('div','',review); nodes.reviewRows.className='vm-review';
        const threeD = section('three-d');
        const datasetFile = input(threeD,'Dataset ZIP / TAR.GZ','file'); datasetFile.accept='.zip,.tar.gz,.tgz';
        const formatLabel=element('label','Dataset ',threeD), datasetFormat=element('select','',formatLabel);
        for(const value of ['nuscenes','kitti']) { const o=element('option',value,datasetFormat); o.value=value; }
        const sourceLabel=element('label','Xử lý ',threeD), datasetMode=element('select','',sourceLabel);
        for(const [value,title] of [['detect','BEVFusion LiDAR + camera'],['import','Nhập nhãn có sẵn']]) {
            const o=element('option',title,datasetMode); o.value=value;
        }
        const uploadScene=input(threeD,'Scene / sequence (trống nếu chỉ có một)','text','');
        const uploadFrames=input(threeD,'Tối đa frame','number','200'); uploadFrames.min='1'; uploadFrames.max='200';
        const uploadInfo=element('p','nuScenes: metadata JSON, samples và sweeps. KITTI: velodyne, ảnh và calib; thêm label_2 hoặc label_02 để nhập nhãn.',threeD);
        let uploadSession3D=null;
        const upload3D=button(threeD,'Upload → xử lý 3D',async()=> {
            const file=datasetFile.files[0];
            if(!file) throw new Error('Chọn dataset ZIP hoặc TAR.GZ');
            const options={filename:file.name,size:file.size,dataset_format:datasetFormat.value,
                mode:datasetMode.value,scene:uploadScene.value.trim(),max_frames:Number(uploadFrames.value)};
            const signature=JSON.stringify(options)+':'+file.lastModified;
            if(!uploadSession3D || uploadSession3D.signature!==signature) {
                const created=await request('/vm/3d/uploads','POST',options);
                uploadSession3D={signature,id:created.job_id,chunk:created.chunk_bytes};
                job=created; localStorage.setItem('autolabel-vm-job',job.job_id);
            }
            let state=await request(`/vm/jobs/${uploadSession3D.id}`);
            for(let offset=state.uploaded_bytes || 0;offset<file.size;) {
                const end=Math.min(offset+uploadSession3D.chunk,file.size);
                let failure;
                for(let attempt=0;attempt<3;attempt++) {
                    try {
                        const response=await fetch(`${api}/vm/3d/uploads/${uploadSession3D.id}?offset=${offset}`,{
                            method:'PUT',credentials:'same-origin',headers:{'X-CSRFToken':csrf(),'Content-Type':'application/octet-stream'},body:file.slice(offset,end)});
                        const payload=await response.json();
                        if(!response.ok) throw new Error(typeof payload.detail==='string'?payload.detail:`HTTP ${response.status}`);
                        state=payload; failure=null; break;
                    } catch(error) { failure=error; if(attempt<2) await new Promise(resolve=>setTimeout(resolve,500)); }
                }
                if(failure) throw failure;
                offset=state.uploaded_bytes;
                uploadInfo.textContent=`Đang upload: ${Math.round(offset/file.size*100)}% (${Math.round(offset/1048576)}/${Math.round(file.size/1048576)} MiB)`;
            }
            job=await request(`/vm/3d/uploads/${uploadSession3D.id}/complete`,'POST');
            uploadSession3D=null; datasetFile.value=''; shown=0;
            uploadInfo.textContent='Đã nhận dataset; đang kiểm tra dữ liệu và tạo task 3D.';
            tab('run'); await poll();
        });
        const syncDatasetMode=()=> {
            const kitti=datasetFormat.value==='kitti';
            datasetMode.options[0].disabled=kitti;
            if(kitti) datasetMode.value='import';
            uploadInfo.textContent=kitti?'KITTI: nhập label_2/label_02, chỉnh sửa trên CVAT và xuất label_2. Auto-label BEVFusion cần checkpoint KITTI riêng.':'nuScenes: BEVFusion cần LiDAR, 6 camera, calibration/ego pose và metadata JSON; có thể chọn nhập sample_annotation.';
        };
        datasetFormat.onchange=syncDatasetMode;
        syncDatasetMode();
        const exportPurpose=element('select','',threeD);
        for(const [value,title] of [['training','Xuất huấn luyện: đã duyệt'],['draft','Xuất bản nháp: có thể chưa duyệt']]) {
            const option=element('option',title,exportPurpose); option.value=value;
        }
        const exportType=element('select','',threeD);
        for(const [value,title] of [['detection','Detection'],['tracking','Tracking (có ID)']]) {
            const option=element('option',title,exportType); option.value=value;
        }
        element('small','nuScenes Tracking gồm 7 lớp; barrier, construction_vehicle và traffic_cone chỉ có trong Detection. KITTI Tracking cần dữ liệu sequence.',threeD);
        nodes.export3d=button(threeD,'Xuất nhãn đã Save',async()=> {
            requireJob();
            if(location.pathname.match(new RegExp(`/jobs/${job.cvat_job_id}$`))) {
                const status=await bridge('status'); if(status.dirty) throw new Error('Save trên CVAT trước khi xuất nhãn');
            }
            const response=await fetch(`${api}/vm/3d/jobs/${job.job_id}/export?purpose=${exportPurpose.value}&annotation_type=${exportType.value}`,{credentials:'same-origin'});
            if(!response.ok) { const payload=await response.json(); throw new Error(payload.detail || `HTTP ${response.status}`); }
            const url=URL.createObjectURL(await response.blob()), link=element('a','',document.body);
            link.href=url; link.download=`${job.dataset_format}-${job.scene}-${exportType.value}-${exportPurpose.value}.${job.dataset_format==='kitti'?'zip':'json'}`;
            link.click(); link.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000);
        }); nodes.export3d.disabled=true;
        const review3DRows=element('div','',threeD);
        let review3DPage=0;
        async function saved3DEditor() {
            requireJob();
            if(job.pipeline!=='lidar3d') throw new Error('Mở một job 3D trước');
            if(location.pathname.match(new RegExp(`/jobs/${job.cvat_job_id}$`))) {
                const status=await bridge('status'); if(status.dirty) throw new Error('Save trước khi duyệt 3D');
            }
        }
        async function review3D() {
            await saved3DEditor();
            const data=await request(`/vm/3d/jobs/${job.job_id}/review?page=${review3DPage}`);
            review3DRows.replaceChildren();
            element('p',`${data.total} object/track · ${data.pending} chờ duyệt · trang ${data.page+1}`,review3DRows);
            data.objects.forEach(object=> {
                const row=element('div',`${object.label} · ${object.track_id || object.object_id} · frame ${object.frame} · ${object.review_status}`,review3DRows);
                button(row,'Đến frame',()=>bridge('jump',{frame:object.frame}));
                for(const [status,title] of [['accepted','Nhận toàn bộ track/object'],['rejected','Bỏ'],['pending','Chờ duyệt']]) {
                    button(row,title,async()=> {
                        await saved3DEditor();
                        await request(`/vm/3d/jobs/${job.job_id}/review`,'POST',{object_id:object.object_id,status,expected_revision:data.revision});
                        if(location.pathname.match(new RegExp(`/jobs/${job.cvat_job_id}$`))) await bridge('sync');
                        await review3D();
                    });
                }
            });
        }
        button(threeD,'Tải danh sách duyệt 3D',async()=> {review3DPage=0;await review3D();});
        button(threeD,'Trang trước',async()=> {review3DPage=Math.max(0,review3DPage-1);await review3D();});
        button(threeD,'Trang sau',async()=> {review3DPage+=1;await review3D();});
        button(threeD,'Xem liên kết 2D–3D ở frame hiện tại',async()=> {
            await saved3DEditor(); const status=await bridge('status');
            const data=await request(`/vm/3d/jobs/${job.job_id}/camera-links?frame=${status.frame}`);
            review3DRows.replaceChildren();
            element('p',`Frame ${data.frame}: ${data.links.filter(link=>link.status==='matched').length} liên kết YOLO/ByteTrack ↔ 3D`,review3DRows);
            for(const link of data.links.filter(link=>link.status==='matched')) {
                element('div',`${link.camera} · ${link.track_3d_id || 'cuboid'} ↔ ${link.track_2d_id} · IoU ${link.iou.toFixed(2)}`,review3DRows);
            }
            element('small','Không ép gán ID khi khác class, khuất camera hoặc nhiều ứng viên gần nhau. Liên kết ghi tại lần chạy model; chỉnh geometry trong CVAT không chạy lại matching.',review3DRows);
        });
        element('p','Chọn scene LiDAR để chạy BEVFusion và tạo task 3D có point cloud, ảnh camera và cuboid chờ duyệt. Bạn cũng có thể tạo task riêng từ PCD/ZIP.',threeD);
        const create3D = element('a','Tạo task point cloud 3D',threeD); create3D.href='/tasks/create';
        const sceneLabel=element('label','Scene LiDAR ',threeD);
        const scene3D=element('select','',sceneLabel);
        const frames3D=input(threeD,'Số frame','number','10'); frames3D.min='1'; frames3D.max='200';
        const ready3D=element('span','Đang kiểm tra detector 3D…',threeD);
        const run3D=button(threeD,'Chạy BEVFusion 3D',async()=> {
            if(!scene3D.value) throw new Error('Chưa có scene LiDAR trên VM');
            job=await request('/vm/3d/jobs','POST',{scene:scene3D.value,max_frames:Number(frames3D.value)});
            localStorage.setItem('autolabel-vm-job',job.job_id); shown=0; await poll();
            ready3D.textContent='Đã xếp hàng chạy 3D. Mở tab Chạy để xem tiến độ và link task.';
        }); run3D.disabled=true;
        request('/vm/3d/capabilities').then(cap=> {
            for(const scene of cap.scenes || []) {
                const option=element('option',`${scene.name} (${scene.frames} frame)`,scene3D);
                option.value=scene.name; option.dataset.frames=String(scene.frames);
            }
            run3D.disabled=!cap.detection_enabled || !scene3D.options.length;
            upload3D.disabled=!cap.enabled;
            datasetMode.options[0].disabled=datasetFormat.value==='kitti' || !cap.detection_enabled;
            if(datasetMode.options[0].disabled) datasetMode.value='import';
            ready3D.textContent=run3D.disabled?'Detector 3D chưa sẵn sàng':'BEVFusion sẵn sàng · tạo task và cuboid chờ duyệt';
            scene3D.onchange=()=> {const max=Number(scene3D.selectedOptions[0]?.dataset.frames || 1); frames3D.max=String(max); frames3D.value=String(Math.min(Number(frames3D.value),max));};
            scene3D.onchange();
        }).catch(()=> {ready3D.textContent='Không đọc được cấu hình 3D; đăng nhập CVAT rồi mở lại panel';});
        const lowMemory = input(threeD,'3D ít RAM (cache 256 MiB, LOD hiển thị)','checkbox');
        lowMemory.checked = localStorage.getItem('cvat.3d.low-memory') === 'true';
        button(threeD,'Áp dụng profile 3D',async()=> {
            await bridge('memory3d',{enabled:lowMemory.checked});
            nodes.notice.textContent=lowMemory.checked?'Đã bật profile 3D ít RAM':'Đã tắt profile 3D ít RAM';
        });
        element('small','Áp dụng khi đang mở job CVAT. Lựa chọn được lưu trên trình duyệt; chỉ ảnh hưởng hiển thị, giữ nguyên nhãn và dữ liệu detector.',threeD);
        const vlm = section('vlm');
        button(vlm,'Áp dụng Settings VLM cho job đã hoàn tất',async()=> {
            requireJob(); await request(`/vm/jobs/${job.job_id}/vlm/profile`,'POST');
            nodes.notice.textContent='Đã áp dụng cấu hình VLM hiện tại cho job này';
        });
        button(vlm,'Hỏi VLM frame hiện tại',async()=> { requireJob(); const status=await bridge('status'); if(status.dirty) throw new Error('Save trước khi hỏi VLM');
            const result=await request(`/vm/jobs/${job.job_id}/vlm`,'POST',{source_frame:status.frame}); renderResult(result); });
        button(vlm,'Đề xuất đã lưu',async()=> { requireJob(); const data=await request(`/vm/jobs/${job.job_id}/vlm`); renderResults(data.results); });
        nodes.results=element('div','',vlm); nodes.results.className='vm-results';
        if(localStorage.getItem('autolabel-vm-collapsed')==='true') panel.classList.add('vm-collapsed');
        tab(localStorage.getItem('autolabel-vm-tab') || 'run');
    }
    function requireJob() { if (!job) throw new Error('Tạo hoặc mở job VM trước'); }
    function removeOverlay() { overlay?.remove(); overlay=null; selected=null; }
    function renderResults(results) {
        nodes.results.replaceChildren();
        if (!results.length) { element('p','Chưa có request VLM hoàn tất',nodes.results); return; }
        const list=element('div','',nodes.results);
        results.forEach(r=>button(list,`Frame ${r.source_frame}: ${r.proposals.length} đề xuất`,()=>renderResult(r)));
    }
    function renderResult(result) {
        nodes.results.replaceChildren(); removeOverlay();
        const image=element('img','',nodes.results); image.src=result.preview; image.alt=`Preview đã che, frame ${result.source_frame}`;
        const rows=element('div','',nodes.results);
        element('p',`Frame ${result.source_frame} · ${result.model} · ${result.reason}`,rows);
        if(!result.proposals.length) element('p','Request thành công, không có đề xuất',rows);
        result.proposals.slice(0,20).forEach((proposal,index)=> {
            const row=element('div',`${proposal.label}: ${proposal.reason} (${proposal.status})`,rows);
            button(row,'Xem box',async()=> { await bridge('jump',{frame:result.source_frame}); selected={frame:result.source_frame,width:result.width,height:result.height,bbox:proposal.bbox}; drawOverlay(); });
            for(const [accept,title] of [[true,'Nhận'],[false,'Bỏ']]) {
                const b=button(row,accept && proposal.status==='accepting'?'Retry Nhận':title,async()=> {
                    const status=await bridge('status'); if(status.dirty) throw new Error('Save trước quyết định VLM');
                    const updated=await request(`/vm/jobs/${job.job_id}/vlm/${result.result_id}/decision`,'POST',{index,accept});
                    renderResult(updated); if(accept) await bridge('sync');
                }); b.disabled=proposal.status!=='pending' && !(accept && proposal.status==='accepting');
            }
        });
        if(result.proposals.length>20) element('p',`${result.proposals.length-20} đề xuất vượt giới hạn hiển thị, giữ trong artifact`,rows);
    }
    async function loadReview() {
        requireJob(); const data=await request(`/vm/jobs/${job.job_id}/review?page=${reviewPage}&class_name=${encodeURIComponent(nodes.reviewClass.value)}&viewed=${encodeURIComponent(nodes.reviewViewed.value)}`);
        nodes.reviewLabel.textContent=`Trang ${data.page}, ${data.total} track (50/trang)`;
        nodes.reviewRows.replaceChildren();
        data.tracks.forEach(track=> {
            const row=element('div',`${track.class} · frame ${track.frame} · ${track.reason} · ${track.state}`,nodes.reviewRows);
            button(row,'Đến object',()=>bridge('jump',{frame:track.frame,trackID:track.track_id}));
            [['seen','Đã xem'],['resolved','Đã xử lý']].forEach(([state,title])=>button(row,title,async()=> {
                await request(`/vm/jobs/${job.job_id}/review/state`,'POST',{track_id:track.track_id,state}); await loadReview();
            }));
        });
    }
    function dialog(title) { const d=element('dialog','',document.body); d.className='vm-dialog'; element('h3',title,d); d.onclose=()=>d.remove(); d.showModal(); return d; }
    async function settingsDialog() {
        const data=await request('/settings/vlm'); const d=dialog('VLM riêng theo tài khoản CVAT');
        element('p',`Đang dùng ${data.model}; key ${data.key_configured?'đã cấu hình':'chưa cấu hình'}. Job đang chạy giữ phiên bản cũ.`,d);
        const base=input(d,'Base URL HTTPS','url',data.endpoint.replace(/\/chat\/completions$/,''));
        const model=input(d,'Model','text',data.model); const key=input(d,'API key riêng (để trống để giữ key riêng hiện tại)','password'); key.autocomplete='new-password';
        const note=element('p','',d); note.setAttribute('role','status');
        async function action(callback) { try{ await callback(); }catch(e){note.textContent=e.message;}finally{key.value='';} }
        button(d,'Lưu',()=>action(async()=> { const body={base_url:base.value,model:model.value}; if(key.value) body.api_key=key.value;
            await request('/settings/vlm','PUT',body); note.textContent='Đã lưu key mã hóa trên server'; }));
        button(d,'Kiểm tra kết nối',()=>action(async()=> { await request('/settings/vlm/test','POST'); note.textContent='Vision API phản hồi schema hợp lệ trên ảnh thử trắng'; }));
        button(d,'Claude qua OpenRouter',()=> {base.value='https://openrouter.ai/api/v1';model.value='anthropic/claude-sonnet-5.5';});
        element('small','Nhập key OpenRouter → Lưu → Kiểm tra kết nối. Với job cũ đã hoàn tất, mở tab VLM và áp dụng Settings cho job đó.',d);
        button(d,'Xóa key riêng',()=>action(async()=> { await request('/settings/vlm','DELETE'); d.close(); }));
        button(d,'Đóng',()=> { key.value=''; d.close(); });
    }
    async function privacyDialog() {
        requireJob(); const d=dialog('Thêm vùng che — tạo video/task mới');
        element('p','Dùng tọa độ pixel trên video đã che. Vùng cố định có thể áp dụng cho một khoảng frame; object chuyển động cần nhiều khoảng/vùng.',d);
        const video=element('video','',d); video.src=api+`/vm/jobs/${job.job_id}/video`; video.controls=true; video.style.width='100%';
        const start=input(d,'Frame đầu','number','0'),end=input(d,'Frame cuối','number','0');
        const box=input(d,'x1,y1,x2,y2 (pixel)','text','');
        const trackLabel=element('label','Track chuyển động (JSON tùy chọn): [{"track_id":"face-1","keyframes":[{"frame":0,"box":[1,2,10,12]},{"frame":20,"box":[20,2,29,12]}]}]',d);
        const tracks=element('textarea','',trackLabel);tracks.style.width='100%';
        button(d,'Tạo bản đã che mới',async()=> { const data=await request(`/vm/jobs/${job.job_id}/privacy-regions`,'POST',{
            regions:box.value.trim()?[{start:Number(start.value),end:Number(end.value),box:box.value.split(',').map(Number)}]:[],
            tracks:tracks.value.trim()?JSON.parse(tracks.value):[]});
            job=data;localStorage.setItem('autolabel-vm-job',job.job_id);shown=0;d.close();await poll(); });
        button(d,'Đóng',()=>d.close());
    }
    window.addEventListener('autolabel:vlm-settings',()=>settingsDialog().catch(e=> { mount();nodes.error.textContent=e.message; }));
    window.addEventListener('autolabel:save-conflict',event=> {
        const {path,local,remote,choose}=event.detail;
        const d=dialog('Hai phiên sửa cùng dữ liệu — chọn giá trị cần giữ');
        element('p',path,d);
        function value(data) {
            const [kind,id,...rest]=path.split('/'); let row=data[kind]?.find(r=>String(r.id)===id);
            for(let i=0;i<rest.length;i++) {
                const part=rest[i];
                if(part==='frames') { row=row?.find(r=>String(r.frame)===rest[++i]); }
                else if(part==='specs') { row=row?.find(r=>String(r.spec_id)===rest[++i]); }
                else row=row?.[part];
            } return row;
        }
        element('p','Chỉnh sửa của bạn',d); element('pre',JSON.stringify(value(local),null,2) ?? 'Đã xóa',d);
        element('p','Phiên trên server',d); element('pre',JSON.stringify(value(remote),null,2) ?? 'Đã xóa',d);
        for(const [choice,title] of [['local','Giữ chỉnh sửa của tôi'],['remote','Giữ phiên trên server'],['cancel','Hủy Save, giữ local']]) {
            button(d,title,()=> { choose(choice); d.close(); });
        }
        d.addEventListener('cancel',()=>choose('cancel'));
    });
    function drawOverlay() {
        const background=document.getElementById('cvat_canvas_background');
        const frameInput=document.querySelector('.cvat-player-frame-selector input');
        if(!selected || !background || frameInput && Number(frameInput.value)!==selected.frame) { overlay?.remove();overlay=null;return; }
        if(!overlay) { overlay=document.createElementNS('http://www.w3.org/2000/svg','svg'); background.parentElement.appendChild(overlay); }
        const css=getComputedStyle(background);
        Object.assign(overlay.style,{position:'absolute',pointerEvents:'none',zIndex:'5',left:css.left,top:css.top,
            width:css.width,height:css.height,transform:css.transform,transformOrigin:css.transformOrigin});
        overlay.setAttribute('viewBox',`0 0 ${selected.width} ${selected.height}`); overlay.replaceChildren();
        const rectangle=document.createElementNS('http://www.w3.org/2000/svg','rect');
        const [a,b,c,d]=selected.bbox;
        Object.entries({x:a*selected.width,y:b*selected.height,width:(c-a)*selected.width,height:(d-b)*selected.height,
            fill:'none',stroke:'red','stroke-width':3,'vector-effect':'non-scaling-stroke'}).forEach(([k,v])=>rectangle.setAttribute(k,String(v)));
        overlay.appendChild(rectangle);
    }
    async function poll() {
        mount();
        if(closed) return;
        const page=document.querySelector('.cvat-annotation-page');
        if(page) { page.classList.add('vm-docked'); if(panel.parentElement!==page) page.appendChild(panel); panel.classList.remove('vm-floating'); }
        else { panel.classList.add('vm-floating'); if(panel.parentElement!==document.body) document.body.appendChild(panel); }
        const path=location.pathname; const match=path.match(/\/tasks\/\d+\/jobs\/(\d+)/);
        if(path!==route) {
            route=path; removeOverlay();
            if(match) { const data=await request(`/vm/cvat-jobs/${match[1]}`); job=data.job; shown=0; }
            else if(!job) { const saved=localStorage.getItem('autolabel-vm-job'); if(saved) job=await request(`/vm/jobs/${saved}`); }
        }
        if(!job) return;
        job=await request(`/vm/jobs/${job.job_id}/processing`);
        nodes.export3d.disabled=job.pipeline!=='lidar3d' || !job.dataset_format || !job.cvat_job_id;
        for(const key of ['privacy_frames','delivered_frames','processed_frames','complete_frames','published_frames']) {
            nodes[key].max=job.total_frames || 1; nodes[key].value=job[key] || 0;
            nodes[key].parentElement.hidden=job.pipeline==='lidar3d' && key==='privacy_frames';
            const title=job.pipeline==='lidar3d' && key==='processed_frames'?'LiDAR có kết quả':({privacy_frames:'Che nhận dạng',delivered_frames:'Tải ảnh',processed_frames:'SAM có kết quả',complete_frames:'Đủ mọi class',published_frames:'Đã gửi CVAT'})[key];
            nodes[key+'Text'].textContent=title+`: ${job[key] || 0}/${job.total_frames || '?'} `;
        }
        nodes.status.textContent=`${job.status} · ${job.stage} · ${job.classes.map(c=>`${c}: ${job.class_coverage?.[c] || 0}`).join(' / ')} · checkpoint ${job.completed_windows || 0} · cache ${Math.round((job.cache_bytes || 0)/1048576)} MiB`;
        if(job.pipeline==='lidar3d') nodes.status.textContent=`3D · ${job.scene} · ${job.status} · ${job.stage} · ${job.cuboid_count || 0} cuboid chờ duyệt`;
        if(job.mask_absence_count) nodes.status.textContent+=` ? ${job.mask_absence_count} mask v?ng/c?n xem`;
        if(job.cap_warning) nodes.status.textContent+=' · Đã chạm cap object; cần review cảnh đông';
        if(job.error) nodes.error.textContent=job.error;
        if(job.cvat_job_id) { nodes.link.href=`/tasks/${job.cvat_task_id}/jobs/${job.cvat_job_id}`; nodes.link.textContent='Mở task đã che'; }
        pending=job.publication_version || 0;
        if(match && pending>shown && !busy) {
            const state=await bridge('status');
            if(state.dirty) nodes.notice.textContent='Có kết quả mới. Save rồi đồng bộ';
            else { await bridge('sync'); shown=pending; nodes.notice.textContent='Đã đồng bộ kết quả mới'; }
        }
        const ctx=nodes.timeline.getContext('2d'),colors=['#999','#e9af20','#287bce','#bd3030'];
        ctx.clearRect(0,0,1000,18);
        (job.frame_states || []).forEach((state,i)=> {ctx.fillStyle=colors[state];ctx.fillRect(i*1000/job.total_frames,0,Math.ceil(1000/job.total_frames),18);});
    }
    launcher.onclick=()=> {closed=false;mount();panel.hidden=false;launcher.hidden=true;poll().catch(e=>{nodes.error.textContent=e.message;});};
    let polling=false;
    setInterval(async()=> {if(polling || closed || !panel) return;polling=true;try{await poll();}catch(e){nodes.error.textContent=e.message;}finally{polling=false;}},2000);
    setInterval(()=> { if(selected) drawOverlay(); },150);
})();
