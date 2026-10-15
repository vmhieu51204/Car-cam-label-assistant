/* AutoLabel controls for CVAT. The browser sends its CVAT session; no service PAT enters this script. */
(async () => {
    'use strict';
    try {
        const response=await fetch('/autolabel/api/v1/vm/capabilities',{credentials:'same-origin'});
        if(response.ok && (await response.json()).enabled) {
            const script=document.createElement('script');script.src='/autolabel-vm.js';
            document.head.appendChild(script);
            return;
        }
    } catch { /* Existing installations keep their previous panel. */ }

    const backend = '/autolabel/api/v1/video/jobs';
    const active = new Set(['queued', 'extracting', 'propagating']);
    const priorityOrder = { critical: 0, high: 1, normal: 2, low: 3 };
    let panel = null;
    let cvatTaskId = null;
    let cvatJobId = null;
    let job = null;
    let queue = null;
    let busy = false;
    let error = '';
    let manifest = null;
    let sourceFps = null;
    let sam3Capabilities = { vlm_box_rescue: false };
    let vlmBoxRescue = false;
    let executionProvider = 'kaggle';
    let modelVariant = 'sam3';
    let boxRescueState = { status: 'disabled', proposals: [] };
    let manualResult = null;
    let manualFrame = null;
    let manualRestoreFrame = null;
    let activeTab = localStorage.getItem('autolabel-tab') || 'run';
    let collapsed = localStorage.getItem('autolabel-collapsed') === '1';
    let triage = null;
    let selectedProposal = null;
    let overlay = null;
    let canvasObserver = null;
    let observedBackground = null;
    let reviewIndex = 0;
    let reviewFilter = 'all';
    let reviewClass = 'all';
    let reviewReason = 'all';
    let reviewViewed = 'all';
    let suspectState = null;
    let showSuspects = false;
    const messages = {
        vi: {run:'Chạy',review:'Duyệt',claude:'Claude',close:'Đóng',collapse:'Thu gọn / mở rộng',
            details:'Thông tin chạy',fps:'FPS video gốc',classes:'Nhãn CVAT',start:'Chạy SAM trên video',
            empty:'Chưa có kết quả. Chạy SAM để tạo nhãn sơ bộ.',loading:'Đang xử lý…',
            pending:'Chờ duyệt',accepted:'Đã nhận',rejected:'Đã bỏ',accepting:'Đang nhận / thử lại',conflict:'Xung đột',
            save:'Lưu chỉnh sửa trước khi tải lại.',reload:'Tải lại CVAT',
            manual:'Hỏi Claude ở frame hiện tại',advanced:'Chọn frame khác (nâng cao)',
            hint:'Đề xuất không phải annotation. Chỉ Nhận mới ghi box vào CVAT.',
            accept:'Nhận',reject:'Bỏ',view:'Xem trên video',none:'Không có đề xuất.',
            full:'Ước lượng cả phần bị che',visible:'Box phần nhìn thấy',
            high:'Cao',medium:'Vừa',low:'Thấp',unseen:'Chưa xem',seen:'Đã xem',resolved:'Đã xử lý',
            prev:'Vấn đề trước',next:'Vấn đề sau',all:'Tất cả',
            importing:'Tải frame',sam:'SAM3',writing:'Ghi CVAT',done:'Hoàn tất',failed:'Lỗi',
            noJob:'Chưa chạy',status:'Trạng thái',saved:'Kết quả đã lưu ở frame',frame:'Frame',
            bulkAccept:'Nhận tất cả đề xuất độ chắc chắn cao',bulkReject:'Bỏ các đề xuất đang chờ',
            busy:'Chờ thao tác hiện tại hoàn tất',wait:'Chờ Claude hoàn tất lượt rà',
            mismatch:'Có chỉnh sửa mới. Hãy Save/Reload rồi thử lại.',
            reasons:{box_jump:'Box thay đổi vị trí mạnh',area_change:'Kích thước thay đổi mạnh',tiny_box:'Vật thể nhỏ',
                short_track:'Track ngắn',track_gap:'Track có khoảng trống',scene_cut:'Gần cảnh cắt',
                stitch_ambiguity:'Ghép ID chưa chắc chắn',ambiguous_overlap:'Overlap khó ghép',unmatched_chunk:'ID mới chưa nối được',
                class_policy:'Nhãn cần kiểm tra',routine_review:'Kiểm tra thông thường',too_short:'Track quá ngắn',
                in_ignore_region:'Trong vùng bỏ qua',edge_truncated:'Sát mép ảnh',low_confidence_label:'Điểm model thấp'}},
        en: {run:'Run',review:'Review',claude:'Claude',close:'Close',collapse:'Collapse / expand',details:'Run details',
            fps:'Source FPS',classes:'CVAT labels',start:'Run SAM on video',empty:'No results yet.',loading:'Working…',
            pending:'Pending',accepted:'Accepted',rejected:'Rejected',accepting:'Accepting / retry',conflict:'Conflict',
            save:'Save edits before reload.',reload:'Reload CVAT',manual:'Ask Claude on current frame',
            advanced:'Other frame (advanced)',hint:'Suggestions are separate until Accept.',accept:'Accept',reject:'Reject',
            view:'Show on video',none:'No suggestions.',full:'Estimated occluded extent',visible:'Visible-part box',
            high:'High',medium:'Medium',low:'Low',unseen:'Unseen',seen:'Seen',resolved:'Resolved',prev:'Previous',next:'Next',all:'All',
            importing:'Frames',sam:'SAM3',writing:'CVAT',done:'Done',failed:'Error',noJob:'Not started',status:'Status',
            saved:'Saved result on frame',frame:'Frame',bulkAccept:'Accept all high-certainty suggestions',
            bulkReject:'Reject pending suggestions',busy:'Wait for current action',wait:'Wait for Claude to finish',
            mismatch:'New edits detected. Save/Reload and retry.', reasons:{}}
    };
    let language = localStorage.getItem('autolabel-language') || 'vi';
    const tr = (key) => messages[language]?.[key] ?? messages.en[key] ?? key;
    const reasonText = (key) => messages[language]?.reasons[key] ?? messages.vi.reasons[key] ?? tr('review');


    function currentSourceFrame() {
        const input = document.querySelector('.cvat-player-frame-selector input');
        const value = Number(input?.value ?? new URL(location.href).searchParams.get('frame') ?? 0);
        return Number.isInteger(value) && value >= 0 ? value : 0;
    }

    function rememberView() {
        const current = document.querySelector('.cvat-player-frame-selector input');
        const frame = Number(current?.value);
        if (current && Number.isInteger(frame) && frame >= 0) {
            const url = new URL(location.href);
            url.searchParams.set('frame', String(frame));
            history.replaceState(history.state, '', url);
        }
        const savedSuggestionFrame = manualResult?.source_frame ?? manualFrame;
        if (cvatJobId && savedSuggestionFrame !== null) {
            sessionStorage.setItem(`autolabel-manual-frame-${cvatJobId}`, String(savedSuggestionFrame));
        }
    }

    function reloadKeepingFrame() {
        rememberView();
        if (panel && cvatJobId) sessionStorage.setItem(`autolabel-reopen-${cvatJobId}`, '1');
        location.reload();
    }

    window.addEventListener('beforeunload', () => {
        rememberView();
        if (panel && cvatJobId) sessionStorage.setItem(`autolabel-reopen-${cvatJobId}`, '1');
    });
    let sequenceClasses = 'car,person';
    let availableClasses = null;
    let classesEdited = false;

    const style = document.createElement('style');
    style.textContent = `
        .autolabel-open { align-self: center; flex-shrink: 0; height: 32px; margin-left: 8px;
            border: 1px solid #1677ff; color: #1677ff; background: white;
            border-radius: 6px; padding: 5px 12px; cursor: pointer; }
        .autolabel-backdrop { position: fixed; inset: 0; z-index: 10000; display: flex;
            justify-content: center; align-items: flex-start; padding-top: 8vh;
            background: rgba(0, 0, 0, .45); }
        .autolabel-panel { width: min(760px, 94vw); max-height: 82vh; overflow: auto;
            padding: 22px 26px; background: white; border-radius: 9px; box-shadow: 0 10px 35px #0003;
            font: 14px/1.5 Arial, sans-serif; color: #222; }
        .autolabel-panel h2 { margin: 0 0 10px; font-size: 21px; }
        .autolabel-panel button { margin: 8px 8px 8px 0; padding: 7px 12px; cursor: pointer;
            border: 1px solid #1677ff; border-radius: 6px; color: white; background: #1677ff; }
        .autolabel-panel button:disabled { opacity: .5; cursor: wait; }
        .autolabel-panel .autolabel-close { float: right; color: #333; border-color: #ccc; background: white; }
        .autolabel-panel .autolabel-error { padding: 9px; color: #9f2222; background: #fff1f0; }
        .autolabel-panel .autolabel-success { padding: 9px; color: #146b38; background: #f0fff2; }
        .autolabel-panel .autolabel-queue { max-height: 240px; overflow: auto; }
        .autolabel-panel .autolabel-queue p { margin: 5px 0; }
        .autolabel-panel .autolabel-suggestion { border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin: 10px 0; }
        .autolabel-panel .autolabel-suggestion img { display: block; max-width: 100%; max-height: 260px; margin: 8px 0; }
        .autolabel-panel progress { width: 100%; height: 12px; }
        .autolabel-panel .autolabel-setting { display: flex; gap: 12px; align-items: center; margin: 10px 0; }
        .autolabel-panel .autolabel-setting select, .autolabel-panel .autolabel-setting input {
            padding: 6px; border: 1px solid #aaa; border-radius: 4px; }
    `;
    document.head.appendChild(style);
    const dockStyle = document.createElement('style');
    dockStyle.textContent = `
        .cvat-annotation-page.autolabel-docked { display:flex!important; flex-direction:column!important; }
        .autolabel-docked > .cvat-annotation-layout-content { height:auto!important; min-height:0; flex:1; }
        .autolabel-dock { flex:0 0 300px; height:300px; min-height:0; width:100%; overflow:hidden; z-index:30; }
        .autolabel-dock.is-collapsed { flex-basis:112px; height:112px; }
        .autolabel-dock .autolabel-panel { box-sizing:border-box; width:100%; max-height:100%; height:100%;
            border-top:2px solid #2955a3; border-radius:0; box-shadow:none; padding:8px 16px; overflow:auto;
            background:var(--autolabel-bg,#fff); color:var(--autolabel-fg,#222); }
        .autolabel-dock h2 { display:none; }
        .autolabel-dock button { background:#174b99; border-color:#174b99; color:#fff; }
        .autolabel-tabs { display:flex; gap:8px; align-items:center; }
        .autolabel-tabs button[aria-selected=true] { background:#173f86; border:2px solid #111; }
        .autolabel-dock button:focus-visible,.autolabel-dock input:focus-visible,.autolabel-dock select:focus-visible,
        .autolabel-dock summary:focus-visible { outline:3px solid #d17600; outline-offset:2px; }
        .autolabel-dock button:disabled { opacity:1; background:#e1e4e8; border-color:#888; color:#414141; cursor:not-allowed; }
        .autolabel-banner { padding:4px 8px; border:1px solid #888; border-radius:4px; }
        .autolabel-tab-body { padding-top:8px; }
        .autolabel-stepper { display:flex; gap:24px; padding-left:20px; }
        .autolabel-stepper [aria-current=step] { font-weight:bold; color:#174b99; text-decoration:underline; }
        .autolabel-chip { padding:3px 8px; border-radius:12px; font-weight:bold; border:1px solid currentColor; }
        .autolabel-chip.high { background:#fff0ef; color:#9a1818; }
        .autolabel-chip.medium { background:#fff4d6; color:#624600; }
        .autolabel-chip.low { background:#edf2fa; color:#243a62; }
        .autolabel-preview { width:min(800px,100%); display:block; }
        .autolabel-dock pre { white-space:pre-wrap; overflow-wrap:anywhere; }
        @media(prefers-color-scheme:dark) { .autolabel-dock { --autolabel-bg:#20242a; --autolabel-fg:#f3f3f3; }
            .autolabel-dock input,.autolabel-dock select { background:#30343a; color:#fff; }
            .autolabel-dock .autolabel-close { color:#fff; background:#30343a; } }
    `;
    document.head.appendChild(dockStyle);

    function el(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        if (className) node.className = className;
        return node;
    }

    async function api(path, options) {
        const requestOptions = { ...options, credentials: 'include' };
        if (requestOptions.method && requestOptions.method.toUpperCase() !== 'GET') {
            const csrf = document.cookie.split('; ').find((part) => part.startsWith('csrftoken='));
            if (csrf) {
                requestOptions.headers = {
                    ...requestOptions.headers,
                    'X-CSRFToken': decodeURIComponent(csrf.slice('csrftoken='.length)),
                };
            }
        }
        const response = await fetch(`${backend}${path}`, requestOptions);
        if (!response.ok) {
            let detail = `AutoLabel HTTP ${response.status}`;
            try { detail = (await response.json()).detail || detail; } catch (_) { /* use status */ }
            const failure = new Error(String(detail));
            failure.status = response.status;
            throw failure;
        }
        return response.json();
    }

    async function refresh() {
        if (!panel) return;
        try {
            job = await api(`/cvat-jobs/${cvatJobId}`);
            if (job.manifest_sha256 && (!manifest || manifest.job_id !== job.job_id)) {
                manifest = await api(`/${job.job_id}/manifest`);
            }
            if (job.status === 'awaiting_review') {
                queue = await api(`/${job.job_id}/review-queue`);
            }
            boxRescueState = await api(`/${job.job_id}/box-rescue`);
            if (sam3Capabilities.track_review && job.track_set_sha256) triage = await api(`/${job.job_id}/review`);
            if (sam3Capabilities.track_filter && job.track_set_sha256) suspectState = await api(`/${job.job_id}/suspect-tracks`);
            if (!manualResult) {
                const query = manualRestoreFrame === null ? '' : `?source_frame=${manualRestoreFrame}`;
                const saved = await api(`/${job.job_id}/manual-claude/latest${query}`);
                if (saved.result) {
                    manualResult = saved.result;
                }
            }
            error = '';
        } catch (failure) {
            if (failure.status === 404) { job = null; queue = null; manifest = null; error = ''; }
            else error = failure.message;
        }
        render();
    }

    async function loadAvailableClasses() {
        try {
            const [names, capabilities] = await Promise.all([
                api(`/cvat-jobs/${cvatJobId}/labels`), api(`/cvat-jobs/${cvatJobId}/capabilities`),
            ]);
            if (!panel) return;
            availableClasses = names;
            sam3Capabilities = capabilities;
            const preferred = ['car', 'person', 'pedestrian', 'truck'];
            const matching = preferred.filter((name) => names.includes(name));
            if (matching.length && !classesEdited) sequenceClasses = matching.join(',');
            configureDock();
            render();
        } catch (failure) {
            if (panel) { error = failure.message; render(); }
        }
    }

    function requestedClasses() {
        const classes = sequenceClasses.split(',').map((value) => value.trim()).filter(Boolean);
        if (!classes.length || new Set(classes).size !== classes.length) {
            throw new Error('Enter unique CVAT class names separated by commas.');
        }
        if (availableClasses) {
            const missing = classes.filter((name) => !availableClasses.includes(name));
            if (missing.length) {
                throw new Error(`Unknown CVAT class: ${missing.join(', ')}. Available: ${availableClasses.join(', ')}.`);
            }
        }
        return classes;
    }

    function classSetting(content) {
        const classesRow = el('label', 'CVAT class names', 'autolabel-setting');
        const classesInput = el('input');
        classesInput.value = sequenceClasses;
        classesInput.addEventListener('input', () => {
            sequenceClasses = classesInput.value;
            classesEdited = true;
        });
        classesRow.appendChild(classesInput);
        content.appendChild(classesRow);
        if (availableClasses) content.appendChild(el('p', `Available in this CVAT job: ${availableClasses.join(', ') || '(none)'}. Use the exact names.`));
    }

    async function operate(action) {
        if (busy) return;
        busy = true;
        error = '';
        render();
        try { await action(); await refresh(); }
        catch (failure) { error = failure.message; }
        finally { busy = false; render(); }
    }

    async function saveCvatAnnotations() {
        const save = document.querySelector('.cvat-annotation-header-save-button');
        if (!save) throw new Error('CVAT Save button is unavailable. Save annotations and retry.');
        save.click();
        await new Promise((resolve) => setTimeout(resolve, 600));
        const deadline = Date.now() + 15000;
        while (save.textContent.includes('Saving') && Date.now() < deadline) {
            await new Promise((resolve) => setTimeout(resolve, 150));
        }
        if (save.textContent.includes('Saving')) throw new Error('CVAT is still saving. Retry after it finishes.');
    }

    function button(text, action) {
        const node = el('button', text);
        node.disabled = busy;
        if (busy) node.title = tr('busy');
        node.addEventListener('click', () => operate(action));
        return node;
    }

    function configureDock() {
        if (!panel || !sam3Capabilities.docked_panel) return;
        const page = document.querySelector('.cvat-annotation-page');
        if (!page) return;
        page.classList.add('autolabel-docked');
        panel.className = `autolabel-dock${collapsed ? ' is-collapsed' : ''}`;
        panel.querySelector('section').setAttribute('role', 'region');
        panel.querySelector('section').removeAttribute('aria-modal');
        page.appendChild(panel);
    }

    function jumpFrame(frame) {
        const input = document.querySelector('.cvat-player-frame-selector input');
        if (!input) throw new Error(tr('failed'));
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        setter.call(input, String(frame));
        input.dispatchEvent(new Event('input', { bubbles: true }));
        input.dispatchEvent(new Event('change', { bubbles: true }));
        input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
    }

    // Normalized source-image coordinates, independent of browser scaling/offset.
    function proposalRect(box, width, height) {
        return { x: box[0] * width, y: box[1] * height,
            width: (box[2] - box[0]) * width, height: (box[3] - box[1]) * height };
    }

    function syncOverlay() {
        if (!selectedProposal || !sam3Capabilities.canvas_overlay || !panel) {
            overlay?.remove(); overlay = null; return;
        }
        const background = document.getElementById('cvat_canvas_background');
        if (!background || currentSourceFrame() !== selectedProposal.frame) {
            overlay?.remove(); overlay = null; return;
        }
        if (!overlay) {
            overlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            overlay.classList.add('autolabel-canvas-proposal');
            background.parentElement.appendChild(overlay);
        }
        const css = getComputedStyle(background);
        if (observedBackground !== background) {
            canvasObserver?.disconnect(); observedBackground = background;
            canvasObserver = new MutationObserver(() => syncOverlay());
            canvasObserver.observe(background, { attributes:true, attributeFilter:['style','width','height'] });
        }
        Object.assign(overlay.style, { position: 'absolute', left: css.left, top: css.top,
            width: css.width, height: css.height, transform: css.transform, transformOrigin: css.transformOrigin,
            pointerEvents: 'none', zIndex: '20' });
        const width = background.width, height = background.height;
        overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
        const signature = JSON.stringify([selectedProposal.frame,selectedProposal.proposal,width,height,busy,language]);
        if (overlay.dataset.signature === signature) return;
        overlay.dataset.signature = signature;
        overlay.replaceChildren();
        const svg = (tag, attributes) => {
            const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
            Object.entries(attributes).forEach(([key, value]) => n.setAttribute(key, String(value)));
            overlay.appendChild(n); return n;
        };
        const p = selectedProposal.proposal;
        svg('rect', { ...proposalRect(p.bbox, width, height), fill: 'none', stroke: '#ff8c00',
            'stroke-width': 3, 'stroke-dasharray': '8 5', 'vector-effect': 'non-scaling-stroke' });
        const label = svg('text', { x: p.bbox[0] * width, y: Math.max(18, p.bbox[1] * height - 5),
            fill: '#fff', stroke: '#111', 'stroke-width': 3, 'paint-order': 'stroke', 'font-size': 18 });
        label.textContent = `${p.label} · ${p.certainty || p.confidence || ''} · ${tr(p.box_extent === 'amodal_estimate' ? 'full' : 'visible')}`;
        if (p.point) svg('circle', { cx: p.point[0] * width, cy: p.point[1] * height, r: 4, fill: '#ff8c00' });
        if (p.status === 'pending' && selectedProposal.ready) {
            const controls = svg('foreignObject', { x: p.bbox[0] * width,
                y: Math.min(height - 40, p.bbox[3] * height), width: 190, height: 40 });
            const body = document.createElementNS('http://www.w3.org/1999/xhtml','div');
            body.style.pointerEvents = 'auto';
            for (const [name, accept] of [['accept',true],['reject',false]]) {
                const action = button(tr(name), async () => {
                    await selectedProposal.decision(accept); selectedProposal = null;
                }); action.style.background = '#173f86'; action.style.color = '#fff'; body.appendChild(action);
            } controls.appendChild(body);
        }
    }

    function showProposal(proposal, frame, decision, waiting = false) {
        selectedProposal = { proposal, frame, decision, ready:!waiting };
        jumpFrame(frame);
        syncOverlay();
    }

    function render() {
        if (!panel) return;
        if (!sam3Capabilities.docked_panel) { legacyRender(); return; }
        configureDock();
        const content = panel.querySelector('.autolabel-content');
        content.replaceChildren();
        const header = panel.querySelector('h2');
        header.textContent = 'AutoLabel';
        panel.querySelector('.autolabel-close').textContent = tr('close');
        const collapse = button(tr('collapse'), async () => {
            collapsed = !collapsed; localStorage.setItem('autolabel-collapsed', collapsed ? '1' : '0');
        });
        content.appendChild(collapse);
        const nav = el('div', undefined, 'autolabel-tabs'); nav.setAttribute('role', 'tablist');
        for (const key of ['run', 'review', 'claude']) {
            const tab = el('button', tr(key)); tab.setAttribute('role', 'tab');
            tab.setAttribute('aria-selected', String(activeTab === key));
            tab.onclick = () => { activeTab = key; localStorage.setItem('autolabel-tab', key); render(); };
            tab.onkeydown = (event) => {
                if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
                event.preventDefault(); const keys = ['run', 'review', 'claude'];
                activeTab = keys[(keys.indexOf(key) + (event.key === 'ArrowRight' ? 1 : 2)) % 3];
                localStorage.setItem('autolabel-tab', activeTab); render();
                panel.querySelector('[aria-selected="true"]').focus();
            };
            nav.appendChild(tab);
        }
        const locale = el('select'); locale.setAttribute('aria-label', 'Language');
        for (const [value, name] of [['vi', 'Tiếng Việt'], ['en', 'English']]) {
            const o = el('option', name); o.value = value; locale.appendChild(o);
        }
        locale.value = language; locale.onchange = () => { language = locale.value; localStorage.setItem('autolabel-language', language); render(); };
        nav.appendChild(locale); content.appendChild(nav);
        const banner = el('div', busy ? tr('loading') : job?.publish_sha256
            ? `${job.published_track_count} track · ${tr('pending')}` : `${tr('status')}: ${job?.status || tr('noJob')}`,
        'autolabel-banner'); banner.setAttribute('role', 'status');
        if (job?.publish_sha256) banner.appendChild(button(tr('reload'), async () => reloadKeepingFrame()));
        content.appendChild(banner);
        if (error || job?.error_message) { const alert = el('p', error || job.error_message, 'autolabel-error'); alert.setAttribute('role', 'alert'); content.appendChild(alert); }
        if (collapsed) return;
        const body = el('div', undefined, 'autolabel-tab-body'); body.setAttribute('role', 'tabpanel');
        content.appendChild(body);
        if (activeTab === 'run') renderRun(body);
        if (activeTab === 'review') renderReview(body);
        if (activeTab === 'claude') renderClaude(body);
        syncOverlay();
    }

    function renderRun(content) {
        const stepper = el('ol', undefined, 'autolabel-stepper');
        const step = !job ? 0 : job.publish_sha256 ? 3 : job.track_set_sha256 ? 2 : job.manifest_sha256 ? 1 : 0;
        ['importing', 'sam', 'writing', 'claude'].forEach((name, i) => {
            const node = el('li', tr(name)); node.setAttribute('aria-current', i === step ? 'step' : 'false'); stepper.appendChild(node);
        }); content.appendChild(stepper);
        if (!job) {
            const fpsRow = el('label', tr('fps'), 'autolabel-setting'); const fps = el('input');
            fps.type = 'number'; fps.min = '0.001'; fps.step = '0.001'; fps.value = sourceFps ?? '';
            fps.oninput = () => { sourceFps = Number(fps.value); }; fpsRow.appendChild(fps); content.appendChild(fpsRow);
            const row = el('label', tr('classes'), 'autolabel-setting'); const input = el('input'); input.value = sequenceClasses;
            input.oninput = () => { sequenceClasses = input.value; classesEdited = true; }; row.appendChild(input); content.appendChild(row);
            if (availableClasses) content.appendChild(el('p', availableClasses.join(', ')));
            const choices = el('select');
            const variants = [['kaggle:sam3', 'Kaggle · SAM3']];
            if (sam3Capabilities.local_sam3) variants.push(['local:sam3', 'Local · SAM3']);
            if (sam3Capabilities.local_sam31) variants.push(['local:sam3.1', 'Local · SAM3.1']);
            variants.forEach(([v, name]) => { const o = el('option', name); o.value = v; choices.appendChild(o); });
            choices.value = `${executionProvider}:${modelVariant}`; content.appendChild(choices);
            content.appendChild(button(tr('start'), async () => {
                if (!Number.isFinite(sourceFps) || sourceFps <= 0) throw new Error(tr('fps'));
                [executionProvider, modelVariant] = choices.value.split(':');
                job = await api(`/cvat-jobs/${cvatJobId}/import`, {method:'POST', headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({strategy:'all_frames',source_fps:sourceFps,auto_start_sequence:true,
                        sequence_classes:requestedClasses(),sam_execution_provider:executionProvider,
                        sam3_model_variant:modelVariant,vlm_box_rescue:!!sam3Capabilities.vlm_box_rescue})});
            }));
        } else {
            if (job.status === 'failed' && /^video_extraction_/.test(job.error_code || ''))
                content.appendChild(button(tr('start'), async () => { job = await api(`/${job.job_id}/retry-import`, {method:'POST'}); }));
            if (job.status === 'failed' && /^sequence_/.test(job.error_code || '')) {
                const row = el('label', tr('classes')); const input = el('input'); input.value = sequenceClasses;
                input.oninput = () => { sequenceClasses = input.value; classesEdited = true; }; row.appendChild(input); content.appendChild(row);
                content.appendChild(button(tr('start'), async () => { job = await api(`/${job.job_id}/sequence/classes`,
                    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(requestedClasses())}); }));
            }
            if (job.status === 'awaiting_seed' && ['uniform','all_frames'].includes(manifest?.sampling_strategy))
                content.appendChild(button(tr('start'), async () => { job = await api(`/${job.job_id}/sequence`, {method:'POST'}); }));
        }
        const details = el('details'); details.appendChild(el('summary', tr('details')));
        details.appendChild(el('pre', JSON.stringify({stage:job?.stage,progress:[job?.progress_current,job?.progress_total],
            sampling:manifest?.sampling_strategy,source_fps:manifest?.source_fps,frames:manifest?.frames.length,
            components:job?.component_versions}, null, 2))); content.appendChild(details);
    }

    function renderReview(content) {
        if (suspectState) {
            content.appendChild(button(`${tr('review')} (${Object.keys(suspectState.suspects).length})`, async () => { showSuspects = !showSuspects; }));
            if (showSuspects) for (const [id, reasons] of Object.entries(suspectState.suspects))
                content.appendChild(el('p', `${id} · ${reasons.map(reasonText).join(' · ')}`));
        }
        if (triage) {
            content.appendChild(el('p', `${triage.seen}/${triage.total} · ${tr('seen')}`));
            const select = el('select'); select.setAttribute('aria-label', tr('review'));
            for (const key of ['all','high','medium','low']) { const o = el('option', tr(key)); o.value = key; select.appendChild(o); }
            select.value = reviewFilter; select.onchange = () => { reviewFilter = select.value; render(); }; content.appendChild(select);
            for (const [values, current, setter, label] of [
                [['all',...new Set(triage.tracks.map(r => r.class))],reviewClass,v => { reviewClass=v; },tr('classes')],
                [['all',...new Set(triage.tracks.flatMap(r => r.reasons))],reviewReason,v => { reviewReason=v; },tr('review')],
                [['all','unseen','seen','resolved'],reviewViewed,v => { reviewViewed=v; },tr('seen')],
            ]) {
                const filter = el('select'); filter.setAttribute('aria-label',label);
                values.forEach(value => { const o=el('option',value === 'all' ? tr('all') : label === tr('classes') ? value : label === tr('seen') ? tr(value) : reasonText(value));
                    o.value=value; filter.appendChild(o); });
                filter.value=current; filter.onchange=() => { setter(filter.value); reviewIndex=0; render(); }; content.appendChild(filter);
            }
            const rows = filteredReviewRows(triage.tracks, reviewFilter, reviewClass, reviewReason, reviewViewed);
            for (const [name, delta] of [['prev',-1],['next',1]]) content.appendChild(button(tr(name), async () => {
                if (!rows.length) return; reviewIndex = (reviewIndex + delta + rows.length) % rows.length;
                jumpFrame(rows[reviewIndex].source_frame);
            }));
            rows.forEach(row => {
                const card = el('section', undefined, 'autolabel-suggestion');
                card.appendChild(el('span', tr(row.level), `autolabel-chip ${row.level}`));
                card.appendChild(el('strong', ` ${row.class} · ${row.track_id}`));
                card.appendChild(el('p', row.reasons.map(reasonText).join(' · ')));
                card.appendChild(button(`${tr('frame')} ${row.source_frame}`, async () => jumpFrame(row.source_frame)));
                for (const [name,state] of [['seen','seen'],['resolved','resolved']]) card.appendChild(button(tr(name), async () => {
                    await api(`/${job.job_id}/review/state`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({track_id:row.track_id,state})});
                })); content.appendChild(card);
            });
        } else if (queue?.units.length) {
            for (const unit of queue.units) {
                const card = el('section', undefined, 'autolabel-suggestion');
                const level = ['critical','high'].includes(unit.priority) ? 'high' : unit.priority === 'normal' ? 'medium' : 'low';
                card.appendChild(el('span', tr(level), `autolabel-chip ${level}`));
                const frame = manifest?.frames[unit.frame_start]?.source_frame_index ?? unit.frame_start;
                card.appendChild(button(`${tr('frame')} ${frame}`, async () => jumpFrame(frame)));
                card.appendChild(el('p', unit.reason_codes.slice(0,3).map(reasonText).join(' · '))); content.appendChild(card);
            }
        } else content.appendChild(el('p', tr('empty')));
    }

    function filteredReviewRows(rows, level, className, reason, state) {
        return rows.filter(r => (level === 'all' || r.level === level) && (className === 'all' || r.class === className)
            && (reason === 'all' || r.reasons.includes(reason)) && (state === 'all' || r.state === state));
    }

    function renderClaude(content) {
        content.appendChild(el('p', `${tr('hint')} ${tr('save')}`));
        if (!job || !sam3Capabilities.vlm_box_rescue) { content.appendChild(el('p', tr('empty'))); return; }
        const ask = async (frame) => {
            manualFrame = frame;
            manualResult = await api(`/${job.job_id}/manual-claude`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_frame:frame})});
            rememberView();
        };
        content.appendChild(button(tr('manual'), async () => ask(currentSourceFrame())));
        const advanced = el('details'); advanced.appendChild(el('summary', tr('advanced')));
        const input = el('input'); input.type = 'number'; input.min = '0'; input.step = '1'; input.value = manualFrame ?? currentSourceFrame();
        input.oninput = () => { manualFrame=Number(input.value); };
        advanced.appendChild(input); advanced.appendChild(button(tr('manual'), async () => ask(Number(input.value)))); content.appendChild(advanced);
        if (manualResult) {
            content.appendChild(el('p', `${tr('saved')} ${manualResult.source_frame} · ${manualResult.reason}`));
            const preview = el('img'); preview.src = manualResult.preview; preview.alt = tr('claude'); preview.className = 'autolabel-preview'; content.appendChild(preview);
            const usage = el('details'); usage.appendChild(el('summary', tr('details'))); usage.appendChild(el('pre', JSON.stringify(manualResult.usage, null, 2))); content.appendChild(usage);
            manualResult.proposals.forEach((proposal, index) => proposalCard(content, proposal, manualResult.source_frame, async accept => {
                manualResult = await api(`/${job.job_id}/manual-claude/${manualResult.result_id}/decision`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index,accept})});
            }));
        }
        if (boxRescueState.error) content.appendChild(el('p', boxRescueState.error, 'autolabel-error'));
        if (boxRescueState.status === 'failed') content.appendChild(button(tr('claude'), async () => {
            boxRescueState = await api(`/${job.job_id}/box-rescue/retry`, {method:'POST'});
        }));
        if (boxRescueState.status === 'done') for (const [key, accept] of [['bulkAccept',true],['bulkReject',false]]) {
            const ids = (boxRescueState.proposals || []).filter(p => p.status === 'pending' && (!accept || p.confidence === 'high')).map(p => p.proposal_id);
            const bulk = button(tr(key), async () => {
                const revision = await api(`/${job.job_id}/box-rescue/revision`);
                const response = await api(`/${job.job_id}/box-rescue/bulk`, {method:'POST',headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({proposal_ids:ids,accept,expected_annotation_revision:revision.revision})});
                const failures = response.results.filter(r => ['failed','conflict'].includes(r.status));
                if (failures.length) throw new Error(failures.map(r => `${r.proposal_id}: ${r.error}`).join('\n'));
            }); bulk.disabled ||= !ids.length; bulk.title = !ids.length ? tr('none') : tr('save'); content.appendChild(bulk);
        }
        if (boxRescueState.truncated_proposal_count) content.appendChild(el('p', `+${boxRescueState.truncated_proposal_count} · ${tr('details')}`));
        for (const p of boxRescueState.proposals || []) {
            const frame = p.source_frame_index ?? manifest?.frames[p.frame_index]?.source_frame_index ?? p.frame_index;
            proposalCard(content, p, frame, async accept => {
                if (!accept) { await api(`/${job.job_id}/box-rescue/${p.proposal_id}/reject`, {method:'POST'}); return; }
                const revision = await api(`/${job.job_id}/box-rescue/revision`);
                await api(`/${job.job_id}/box-rescue/${p.proposal_id}/accept`, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_annotation_revision:revision.revision})});
            }, boxRescueState.status !== 'done');
        }
        if (!manualResult?.proposals.length && !boxRescueState.proposals?.length) content.appendChild(el('p', tr('none')));
    }

    function proposalCard(content, proposal, frame, decision, waiting = false) {
        const card = el('section', undefined, 'autolabel-suggestion');
        card.appendChild(el('strong', `${proposal.label} · ${tr('frame')} ${frame} · ${tr(proposal.status)}`));
        card.appendChild(el('p', proposal.reason));
        card.appendChild(el('p', tr(proposal.box_extent === 'amodal_estimate' ? 'full' : 'visible')));
        card.appendChild(button(tr('view'), async () => showProposal(proposal, frame, decision, waiting)));
        if (['pending','accepting'].includes(proposal.status)) for (const [name, accept] of [['accept',true],['reject',false]]) {
            const action = button(tr(name), async () => {
                try { await decision(accept); selectedProposal = null; }
                catch (failure) { if (failure.status === 409) throw new Error(tr('mismatch')); throw failure; }
            });
            action.disabled ||= waiting || (!accept && proposal.status === 'accepting');
            if (action.disabled) action.title = tr(waiting ? 'wait' : 'busy'); card.appendChild(action);
        } content.appendChild(card);
    }

    function legacyRender() {
        if (!panel) return;
        const content = panel.querySelector('.autolabel-content');
        content.replaceChildren();
        if (error) content.appendChild(el('p', error, 'autolabel-error'));
        content.appendChild(el('p', 'AutoLabel tracks objects through the video and ranks frames for human review.'));
        content.appendChild(el('p', 'SAM processes every source frame on the selected GPU backend. Legacy jobs retain their original sampling.'));
        if (!job) {
            content.appendChild(el('p', 'SAM3 observes every source frame. All labels remain preliminary and require review.'));
            const fpsRow = el('label', 'CVAT source FPS', 'autolabel-setting');
            const fpsInput = el('input');
            fpsInput.type = 'number';
            fpsInput.min = '0.001';
            fpsInput.step = '0.001';
            fpsInput.placeholder = 'Enter the actual video FPS';
            fpsInput.value = sourceFps === null ? '' : String(sourceFps);
            fpsInput.addEventListener('change', () => { sourceFps = Number(fpsInput.value); });
            fpsRow.appendChild(fpsInput);
            content.appendChild(fpsRow);
            if (sam3Capabilities.vlm_box_rescue) {
                vlmBoxRescue = true;
                content.appendChild(el('p', 'Claude Sonnet reviews at most 20 frames and offers at most 20 boxes. Occluded-object boxes estimate full extent and require human review. Suggestions stay separate until Accept.'));
            }
            classSetting(content);
            const choices = el('select');
            const variants = [['kaggle:sam3', 'Kaggle · SAM3']];
            if (sam3Capabilities.local_sam3) variants.push(['local:sam3', 'Local GPU · SAM3']);
            if (sam3Capabilities.local_sam31) variants.push(['local:sam3.1', 'Local GPU · SAM3.1 (experimental)']);
            variants.forEach(([value, label]) => {
                const option = el('option', label); option.value = value; choices.appendChild(option);
            });
            choices.value = `${executionProvider}:${modelVariant}`;
            if (!choices.value) choices.value = 'kaggle:sam3';
            choices.onchange = () => { [executionProvider, modelVariant] = choices.value.split(':'); };
            content.appendChild(choices);
            content.appendChild(button('Run selected SAM on full video', async () => {
                [executionProvider, modelVariant] = choices.value.split(':');
                const classes = requestedClasses();
                if (!Number.isFinite(sourceFps) || sourceFps <= 0) {
                    throw new Error('Enter the actual source FPS.');
                }
                job = await api(`/cvat-jobs/${cvatJobId}/import`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ strategy: 'all_frames', source_fps: sourceFps,
                        auto_start_sequence: true, sequence_classes: classes,
                        sam_execution_provider: executionProvider, sam3_model_variant: modelVariant,
                        vlm_box_rescue: vlmBoxRescue }),
                });
            }));
            return;
        }
        if (sam3Capabilities.vlm_box_rescue) {
            content.appendChild(el('h3', 'Ask Claude on a selected frame'));
            content.appendChild(el('p', 'Save CVAT edits first. Blue: current labels; red: Claude suggestions. Nothing is written until Accept.'));
            const frameInput = el('input');
            frameInput.type = 'number'; frameInput.min = '0'; frameInput.step = '1';
            frameInput.style.width = '100px';
            const current = document.querySelector('.cvat-player-frame-selector input');
            frameInput.value = String(manualFrame ?? Number(current?.value || 0));
            frameInput.oninput = () => { manualFrame = Number(frameInput.value); };
            content.appendChild(el('label', 'Source frame: '));
            content.appendChild(frameInput);
            content.appendChild(button('Use current CVAT frame', async () => {
                manualFrame = currentSourceFrame();
            }));
            content.appendChild(button('Claude: suggest boxes on this frame', async () => {
                manualFrame = Number(frameInput.value);
                if (!Number.isInteger(manualFrame) || manualFrame < 0) throw new Error('Enter a valid source frame.');
                manualResult = await api(`/${job.job_id}/manual-claude`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source_frame: manualFrame }),
                });
                rememberView();
            }));
            if (manualResult) {
                content.appendChild(el('p', `Saved result for frame ${manualResult.source_frame}: ${manualResult.proposals.length} suggestion(s). ${manualResult.reason}`));
                if (manualFrame !== manualResult.source_frame) {
                    content.appendChild(el('p', `Selected frame: ${manualFrame}. Click Claude to request boxes for this frame; the preview below is the saved result for frame ${manualResult.source_frame}.`));
                }
                const preview = el('img'); preview.src = manualResult.preview; preview.style.width = '100%';
                content.appendChild(preview);
                manualResult.proposals.forEach((proposal, index) => {
                    content.appendChild(el('p', `${index + 1}. ${proposal.label}: ${proposal.reason} (${proposal.status})`));
                    if (proposal.status === 'pending') {
                        for (const [title, accept] of [['Accept box', true], ['Reject box', false]]) {
                            content.appendChild(button(title, async () => {
                                manualResult = await api(`/${job.job_id}/manual-claude/${manualResult.result_id}/decision`, {
                                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ index, accept }),
                                });
                            }));
                        }
                    }
                });
                content.appendChild(el('p', 'After accepting, save any other edits and reload CVAT to see the new track. If revision conflicts, save and ask Claude again.'));
            }
        }
        if (manifest && !['uniform', 'all_frames'].includes(manifest.sampling_strategy)) {
            content.appendChild(el('p', 'This job uses the retired frame-0 workflow. Existing annotations remain in CVAT; use the legacy API to recover an unfinished job.'));
            content.appendChild(el('p', `Legacy job status: ${job.status}`));
            return;
        }
        const runningSequence = job.stage === 'sequence_running' && job.status === 'propagating';
        content.appendChild(el('p', runningSequence
            ? `Status: SAM running · ${job.progress_total} input frames · waiting for batch results`
            : `Status: ${job.stage} · ${job.progress_current}/${job.progress_total} frames`));
        if (['uniform', 'all_frames'].includes(manifest?.sampling_strategy)) {
            content.appendChild(el('p', manifest.sampling_strategy === 'all_frames'
                ? `Original video: ${manifest.source_frame_count} frames at ${manifest.source_fps} FPS. SAM3 processes all ${manifest.frames.length} source frames.`
                : `Legacy sampled job: ${manifest.frames.length}/${manifest.source_frame_count} frames at ${manifest.target_fps} FPS.`));
            content.appendChild(el('p', 'Preliminary labels awaiting human review. Missing observations are not model predictions; legacy sampled jobs use geometric interpolation. Review warnings are not confirmed errors.'));
        } else if (job.status === 'extracting') {
            content.appendChild(el('p', `Preparing input frames: ${job.progress_current}/${job.progress_total}. This is frame preparation, not SAM3 inference.`));
        }
        const progress = el('progress');
        progress.max = job.progress_total || 1;
        if (!runningSequence) progress.value = job.progress_current;
        content.appendChild(progress);
        if (manifest) {
            content.appendChild(el('p', `${manifest.frames.length} selected / ${manifest.source_frame_count || manifest.frames.length} CVAT frames · ${manifest.sampling_strategy || 'all frames'}`));
        }
        if (job.error_message) content.appendChild(el('p', job.error_message, 'autolabel-error'));
        if (job.status === 'failed' && /^video_extraction_/.test(job.error_code || '')) {
            content.appendChild(button('Retry frame import', async () => {
                job = await api(`/${job.job_id}/retry-import`, { method: 'POST' });
            }));
        }
        const sequenceJob = ['uniform', 'all_frames'].includes(manifest?.sampling_strategy);
        if (job.status === 'awaiting_seed' && sequenceJob) {
            content.appendChild(el('p', 'Frame extraction finished. Start or resume SAM3 text tracking without a frame-0 seed.'));
            content.appendChild(button('Start SAM3 sequence', async () => {
                job = await api(`/${job.job_id}/sequence`, { method: 'POST' });
            }));
        }
        if (job.status === 'propagating' && sequenceJob) {
            content.appendChild(el('p', 'SAM is processing the video using the provider saved with this job. CVAT retains every source frame.'));
        }
        if (job.status === 'failed' && /^sequence_/.test(job.error_code || '')) {
            classSetting(content);
            content.appendChild(button('Retry SAM3 with these classes', async () => {
                job = await api(`/${job.job_id}/sequence/classes`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestedClasses()),
                });
            }));
            content.appendChild(el('p', 'This retry reuses the frames already prepared; it does not upload or extract the video again.'));
        }
        if (job.status === 'awaiting_review' && !job.publish_sha256 &&
            ['sequence_draft_ready', 'sequence_draft_exporting'].includes(job.stage)) {
            content.appendChild(el('p', 'SAM3 finished. The worker is adding pending tracks to CVAT automatically.'));
        } else if (job.status === 'awaiting_review' && !job.publish_sha256) {
            content.appendChild(button('Add AI tracks to this CVAT job', async () => {
                await saveCvatAnnotations();
                await api(`/${job.job_id}/publish`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ expected_annotation_revision: null }),
                });
                reloadKeepingFrame();
            }));
        }
        if (job.publish_sha256) {
            content.appendChild(el('p', `${job.published_track_count} AI track(s) added to CVAT`, 'autolabel-success'));
            content.appendChild(el('p', 'Save any edits you are making in CVAT before reloading to view the new tracks.'));
            content.appendChild(button('Reload CVAT to see AI tracks', async () => { reloadKeepingFrame(); }));
        }
        if (boxRescueState && boxRescueState.status !== 'disabled') {
            content.appendChild(el('h3', 'Claude Sonnet box suggestions'));
            if (boxRescueState.completed_frame_count) {
                content.appendChild(el('p', `Saved results from ${boxRescueState.completed_frame_count} completed frame(s).`));
            }
            content.appendChild(el('p', `Status: ${boxRescueState.status} · ${boxRescueState.proposal_count || 0} proposal(s). These are suggestions, not confirmed objects.`));
            if (boxRescueState.truncated_proposal_count) {
                content.appendChild(el('p', `${boxRescueState.truncated_proposal_count} additional suggestions were saved in frame artifacts but omitted from the 20-proposal review limit.`));
            }
            if (boxRescueState.error) content.appendChild(el('p', boxRescueState.error, 'autolabel-error'));
            if (boxRescueState.status === 'failed') {
                content.appendChild(button('Retry Claude Sonnet review (no SAM rerun)', async () => {
                    boxRescueState = await api(`/${job.job_id}/box-rescue/retry`, { method: 'POST' });
                }));
            }
            for (const proposal of (boxRescueState.proposals || [])) {
                const card = el('section', undefined, 'autolabel-suggestion');
                const sourceFrame = manifest?.frames[proposal.frame_index]?.source_frame_index ?? proposal.source_frame_index;
                const link = el('a', `Source frame ${sourceFrame}`);
                link.href = `/tasks/${cvatTaskId}/jobs/${cvatJobId}?frame=${sourceFrame}`;
                card.appendChild(link);
                card.appendChild(el('p', `${proposal.label} · confidence ${proposal.confidence} · ${proposal.occlusion} occlusion · [${proposal.bbox.map((v) => v.toFixed(3)).join(', ')}] normalized`));
                card.appendChild(el('p', proposal.reason));
                card.appendChild(el('p', proposal.box_extent === 'amodal_estimate'
                    ? 'Estimated full object extent, including occluded parts. Check geometry before accepting.'
                    : 'Legacy visible-part box.'));
                const image = el('img');
                image.src = `${backend}/${job.job_id}/box-rescue/${proposal.proposal_id}/preview`;
                image.alt = `SAM boxes and Claude Sonnet proposal for frame ${sourceFrame}`;
                card.appendChild(image);
                if (['pending', 'accepting'].includes(proposal.status)) {
                    const acceptButton = button(proposal.status === 'accepting' ? 'Retry Accept (outcome unknown)' : 'Accept as CVAT track', async () => {
                        await saveCvatAnnotations();
                        const annotations = await api(`/${job.job_id}/box-rescue/revision`);
                        await api(`/${job.job_id}/box-rescue/${proposal.proposal_id}/accept`, {
                            method: 'POST', headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ expected_annotation_revision: annotations.revision }),
                        });
                        error = 'Accepted suggestion was added as a one-frame CVAT track. Reload to display it.';
                    });
                    const rejectButton = button('Reject', async () => {
                        await api(`/${job.job_id}/box-rescue/${proposal.proposal_id}/reject`, { method: 'POST' });
                    });
                    acceptButton.disabled ||= boxRescueState.status !== 'done';
                    rejectButton.disabled ||= boxRescueState.status !== 'done' || proposal.status === 'accepting';
                    card.append(acceptButton, rejectButton);
                    if (boxRescueState.status !== 'done') {
                        card.appendChild(el('p', 'Review actions become available after the Claude Sonnet pass finishes.'));
                    }
                } else {
                    const reviewed = ['accepted', 'rejected'].includes(proposal.status);
                    card.appendChild(el('p', `Status: ${proposal.status}`, reviewed ? 'autolabel-success' : 'autolabel-error'));
                }
                content.appendChild(card);
            }
            if (boxRescueState.proposals?.some((item) => item.status === 'accepted')) {
                content.appendChild(button('Reload CVAT to see accepted suggestions', async () => { reloadKeepingFrame(); }));
            }
        }
        if (queue && queue.units.length) {
            content.appendChild(el('h3', 'Review queue'));
            const list = el('div', undefined, 'autolabel-queue');
            [...queue.units].sort((a, b) =>
                (priorityOrder[a.priority] ?? 9) - (priorityOrder[b.priority] ?? 9) ||
                (b.review_score ?? 0) - (a.review_score ?? 0) || a.frame_start - b.frame_start
            ).forEach((unit) => {
                const row = el('p');
                const sourceStart = manifest?.frames[unit.frame_start]?.source_frame_index ?? unit.frame_start;
                const sourceEnd = manifest?.frames[unit.frame_end]?.source_frame_index ?? unit.frame_end;
                const link = el('a', `Frame ${sourceStart}${sourceEnd > sourceStart ? `â€“${sourceEnd}` : ''}`);
                link.href = `/tasks/${cvatTaskId}/jobs/${cvatJobId}?frame=${sourceStart}`;
                row.append(link, document.createTextNode(` · ${unit.priority} · ${unit.reason_codes.join(', ')}`));
                list.appendChild(row);
            });
            content.appendChild(list);
        }
    }

    function openPanel(taskId, jobId) {
        if (panel) closePanel();
        cvatTaskId = taskId;
        cvatJobId = jobId;
        job = null;
        queue = null;
        manifest = null;
        error = '';
        availableClasses = null;
        boxRescueState = { status: 'disabled', proposals: [] };
        manualResult = null;
        triage = null; suspectState = null; selectedProposal = null;
        reviewFilter='all'; reviewClass='all'; reviewReason='all'; reviewViewed='all'; reviewIndex=0;
        const savedManualFrame = sessionStorage.getItem(`autolabel-manual-frame-${jobId}`);
        manualRestoreFrame = savedManualFrame === null ? null : Number(savedManualFrame);
        manualFrame = currentSourceFrame();
        classesEdited = false;
        panel = el('div', undefined, 'autolabel-backdrop');
        const dialog = el('section', undefined, 'autolabel-panel');
        dialog.setAttribute('role', 'dialog');
        dialog.setAttribute('aria-modal', 'true');
        dialog.setAttribute('aria-label', 'AutoLabel video');
        const close = el('button', 'Close', 'autolabel-close');
        close.addEventListener('click', closePanel);
        dialog.append(close, el('h2', 'AutoLabel video'), el('div', undefined, 'autolabel-content'));
        panel.appendChild(dialog);
        panel.addEventListener('click', (event) => { if (event.target === panel) closePanel(); });
        document.body.appendChild(panel);
        configureDock();
        render();
        refresh();
        loadAvailableClasses();
    }

    function closePanel() {
        rememberView();
        if (cvatJobId) sessionStorage.removeItem(`autolabel-reopen-${cvatJobId}`);
        if (panel) panel.remove();
        document.querySelector('.cvat-annotation-page')?.classList.remove('autolabel-docked');
        overlay?.remove(); overlay = null; selectedProposal = null;
        canvasObserver?.disconnect(); canvasObserver = null; observedBackground = null;
        panel = null;
    }

    function mountButton() {
        const match = location.pathname.match(/^\/tasks\/(\d+)\/jobs\/(\d+)/);
        if (panel && (!match || Number(match[2]) !== cvatJobId)) closePanel();
        const leftGroup = document.querySelector('.cvat-annotation-header-left-group');
        if (!match || !leftGroup) return;
        const existing = document.querySelector('.autolabel-open');
        if (existing) {
            if (existing.parentElement !== leftGroup) leftGroup.appendChild(existing);
            return;
        }
        const launch = el('button', 'AutoLabel video', 'autolabel-open');
        launch.addEventListener('click', () => {
            const current = location.pathname.match(/^\/tasks\/(\d+)\/jobs\/(\d+)/);
            if (current) openPanel(Number(current[1]), Number(current[2]));
        });
        leftGroup.appendChild(launch);
    }

    window.setInterval(() => {
        syncOverlay();
        mountButton();
        if (panel && job && (active.has(job.status) || ['queued', 'running'].includes(boxRescueState.status)) && !busy &&
            !(panel.contains(document.activeElement) && ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))) refresh();
    }, 1800);
    // Scoped combinations: never steal plain CVAT shortcuts or typing in an input.
    document.addEventListener('keydown', event => {
        if (!panel || !panel.contains(document.activeElement) || !event.altKey || !event.shiftKey || busy) return;
        if (['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)) return;
        const issues = filteredReviewRows(triage?.tracks || [],reviewFilter,reviewClass,reviewReason,reviewViewed);
        if (['ArrowUp','ArrowDown'].includes(event.key) && activeTab === 'review' && issues.length) {
            event.preventDefault(); event.stopPropagation();
            reviewIndex = (reviewIndex + (event.key === 'ArrowDown' ? 1 : issues.length - 1)) % issues.length;
            jumpFrame(issues[reviewIndex].source_frame);
        }
        if (['a','r'].includes(event.key.toLowerCase()) && selectedProposal?.ready && selectedProposal.proposal.status === 'pending') {
            event.preventDefault(); event.stopPropagation();
            operate(async () => { await selectedProposal.decision(event.key.toLowerCase() === 'a'); selectedProposal=null; });
        }
    });
    mountButton();
    const current = location.pathname.match(/^\/tasks\/(\d+)\/jobs\/(\d+)/);
    if (current && sessionStorage.getItem(`autolabel-reopen-${current[2]}`)) {
        sessionStorage.removeItem(`autolabel-reopen-${current[2]}`);
        openPanel(Number(current[1]), Number(current[2]));
    }
})();
