const $ = (s)=>document.querySelector(s);
const PALETTE = ['#4f9dff','#ff8a5c','#5ad19a','#c98bff','#ffd24a','#ff6b9d','#6be0e0','#b0b85a','#e0846b','#8a9bff','#7ad17a','#d99bff'];
const esc = (s)=> (s==null?'':String(s)).replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt = (x)=> (x==null||isNaN(x)) ? '–' : Number(x).toFixed(2);
const NUL = '\x1f';
const SVGNS = 'http://www.w3.org/2000/svg';
const svgEl = (t,a,txt)=>{ const e=document.createElementNS(SVGNS,t); for(const k in (a||{})) e.setAttribute(k,a[k]); if(txt!=null) e.textContent=txt; return e; };
const elh = (t,a,txt)=>{ const e=document.createElement(t); for(const k in (a||{})) e.setAttribute(k,a[k]); if(txt!=null) e.textContent=txt; return e; };
const mean = (xs)=> xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;

const cidx = {}; FAM.models.forEach((m,i)=> cidx[m]=i);
const mcolor = (m)=> PALETTE[(cidx[m]||0) % PALETTE.length];
// group color for the pooled-judge tint on each response (-1/absent => neutral grey)
const gcolor = (g)=> (g==null||g<0) ? '#555' : PALETTE[g % PALETTE.length];
const pstd = (xs)=>{ if(xs.length<2) return 0; const m=mean(xs); return Math.sqrt(xs.reduce((s,x)=>s+(x-m)*(x-m),0)/xs.length); };

// Cohorts: our fine-tuned organisms vs base/frontier models (classified server-side).
const COHORTS = ['finetuned','base'];
const COHORT_LABEL = {finetuned:'fine-tuned organisms', base:'base / frontier'};
const COHORT_COLOR = {finetuned:'#ff8a5c', base:'#4f9dff'};
const cohortOf = (m)=> (FAM.cohorts||{})[m] || 'base';
const hasCohorts = ()=> FAM.models.some(m=>cohortOf(m)==='finetuned') && FAM.models.some(m=>cohortOf(m)==='base');

// cross-variant metrics — the ones that apply to the judge that saw ALL variants.
const METRICS = [
  {key:'judge_ari',   label:'judge ARI (framing split)'},
  {key:'swing_norm',  label:'scalar swing (normalized)'},
  {key:'cluster_ari', label:'embedding cluster ARI'},
  {key:'contradiction', label:'judge contradiction (0/1)'},
];
const mlabel = (k)=> (METRICS.find(m=>m.key===k)||{}).label || k;

const REPORT_ID = (FAM.run_dir || 'families') + (FAM.judge_run ? '::'+FAM.judge_run : '');
const SKEY = 'families_report_state_v1:' + REPORT_ID;
const DEFAULTS = { model:'__all__', family:'__all__', sort:'judge_ari', search:'',
  onlyContra:false, metric:'judge_ari', cmode:'cohort', cmodels:null /* null => all */ };
let STATE = Object.assign({}, DEFAULTS);
const openCards = new Set();   // cardKey of expanded bundles
const openResps = new Set();   // respKey of expanded responses
let focusKey = null;           // a card the chart pointed us at (highlight)

const cardKey = (r)=> r.model + NUL + r.family;
const respKey = (r,vi,i)=> cardKey(r) + NUL + vi + NUL + i;

function loadState(){
  let s = {};
  try { s = JSON.parse(localStorage.getItem(SKEY) || '{}'); } catch(e) {}
  const h = new URLSearchParams(location.hash.slice(1));
  for (const k of ['model','family','sort','search','metric','cmode']) if (h.has(k)) s[k]=h.get(k);
  if (h.has('onlyContra')) s.onlyContra = h.get('onlyContra')==='1';
  return Object.assign({}, DEFAULTS, s);
}
function saveState(){
  try { localStorage.setItem(SKEY, JSON.stringify(STATE)); } catch(e) {}
  const h = new URLSearchParams();
  for (const k of ['model','family','sort','search','metric','cmode']) if (STATE[k]!==DEFAULTS[k]) h.set(k, STATE[k]);
  if (STATE.onlyContra) h.set('onlyContra','1');
  const hs = h.toString();
  history.replaceState(null, '', hs ? ('#'+hs) : (location.pathname+location.search));
}

// ---- chart model selection (Set of model names; null === all) ----
function selModels(){ return STATE.cmodels ? new Set(STATE.cmodels) : new Set(FAM.models); }
function setSelModels(set){ STATE.cmodels = (set.size===FAM.models.length) ? null : [...set]; }

// ---- filtering / sorting of bundle cards ----
function passes(r){
  if (STATE.model!=='__all__' && r.model!==STATE.model) return false;
  if (STATE.family!=='__all__' && r.family!==STATE.family) return false;
  if (STATE.onlyContra){
    const j = r.judge||{};
    if (!(j.contradiction || (j.ari??0) >= 0.2)) return false;
  }
  if (STATE.search){
    const q = STATE.search.toLowerCase();
    const hay = [r.model, r.family, (r.judge||{}).rationale||'',
      ...((r.judge||{}).flags||[]),
      ...r.variants.flatMap(v=>v.responses||[])].join('\n').toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}
function sortRecords(rows){
  const ari=(r)=>(r.judge||{}).ari, sw=(r)=>r.swing, contra=(r)=>(r.judge||{}).contradiction?1:0;
  const num=(x)=> (x==null?-1:x);
  const tie=(a,b)=> a.family.localeCompare(b.family) || a.model.localeCompare(b.model);
  const cmps = {
    judge_ari:(a,b)=> num(ari(b))-num(ari(a)) || tie(a,b),
    swing:(a,b)=> num(sw(b))-num(sw(a)) || tie(a,b),
    contradiction:(a,b)=> contra(b)-contra(a) || num(ari(b))-num(ari(a)) || tie(a,b),
    model:(a,b)=> a.model.localeCompare(b.model) || a.family.localeCompare(b.family),
    family:(a,b)=> a.family.localeCompare(b.family) || a.model.localeCompare(b.model),
  };
  return rows.slice().sort(cmps[STATE.sort] || tie);
}
const swingCls = (kind, x)=>{ if (x==null) return 'g-mut';
  if (kind==='number') return x<1.0?'g-green':x<3.0?'g-amber':'g-red';
  return x<0.15?'g-green':x<0.40?'g-amber':'g-red'; };
const ariCls = (x)=> x==null ? 'g-mut' : (x<0.10?'g-green':x<0.40?'g-amber':'g-red');

// ---- bundle card ----
function renderCard(r){
  const key = cardKey(r), isOpen = openCards.has(key), j = r.judge||{}, kind = r.scalar_kind;
  let dots = '';
  if (j.contradiction) dots += '<span class="dot red" title="pooled judge: contradiction"></span>';
  if (j.flags && j.flags.length) dots += '<span class="dot amber" title="flagged"></span>';
  const fmeta = FAM.families[r.family] || {};
  let h = '<div class="card'+(isOpen?' open':'')+(focusKey===key?' focus':'')+'" data-key="'+esc(key)+'">';
  h += '<div class="card-head" data-card="'+esc(key)+'">'
     + '<span class="chev">▶</span>'
     + '<span class="tag model">'+esc(r.model)+'</span>'
     + '<span class="tag fam">'+esc(r.family)+'</span>'
     + '<span class="dots">'+dots+'</span>'
     + '<span class="stats">'
       + '<span>swing <span class="pill '+swingCls(kind,r.swing)+'">'+fmt(r.swing)+'</span></span>'
       + '<span>judge ARI <span class="pill '+ariCls(j.ari)+'">'+fmt(j.ari)+'</span></span>'
       + '<span>cluster ARI <span class="pill '+ariCls((r.cluster||{}).ari)+'">'+fmt((r.cluster||{}).ari)+'</span></span>'
       + '<span>judge grps <b>'+(j.n_groups??'–')+'</b></span>'
     + '</span></div>';
  if (isOpen){
    h += '<div class="body">';
    h += '<div class="q">'+esc(fmeta.prompt || (fmeta.description||r.family))+'</div>';
    // pooled-judge takeaway (the judge that saw every variant at once)
    h += '<div class="takeaway"><div class="lab">pooled judge · saw all '+r.variants.length+' framings</div>';
    if (j.rationale) h += '<div class="rationale">'+esc(j.rationale)+'</div>';
    if (j.flags && j.flags.length)
      h += '<div class="flags">'+j.flags.map(f=>'<span class="flag">'+esc(f)+'</span>').join('')+'</div>';
    h += contingencyHtml(r) + '</div>';
    // group legend (if the judge labels were persisted, the columns are tinted)
    const ng = j.n_groups||0;
    const tinted = r.variants.some(v=>(v.groups||[]).some(g=>g!=null));
    if (ng && tinted){
      const cont = j.contingency||[], gids = j.group_ids||[];
      let leg = '';
      for (let gi=0; gi<ng; gi++){
        const g = gids[gi] ?? gi;
        const cnt = cont.reduce((s,row)=>s+((row||[])[gi]||0), 0);
        leg += '<span class="sw"><span class="box" style="background:'+gcolor(g)+'"></span>judge g'+g+(cnt?' · '+cnt:'')+'</span>';
      }
      leg += '<span class="cav">'+(r.groups_exact
        ? 'response tints are exact per response'
        : 'tints recovered from counts — split variants stay grey; see the matrix')+'</span>';
      h += '<div class="legendrow">'+leg+'</div>';
    } else if (ng > 1 && !r.groups_exact){
      h += '<div class="legendrow"><span class="cav">judge groups shown in the matrix only (this run predates per-response labels)</span></div>';
    }
    // columns: one per framing variant
    h += '<div class="cols">';
    r.variants.forEach((v,vi)=> h += renderColumn(r, v, vi));
    h += '</div></div>';
  }
  h += '</div>';
  return h;
}

function renderColumn(r, v, vi){
  let h = '<div class="col"><div class="col-head">'
    + '<span class="vname">'+esc(v.variant)+'</span>'
    + '<span class="vsum">'+esc(v.summary)+' · n='+(v.responses||[]).length+'</span></div>';
  h += '<div class="col-body">';
  (v.responses||[]).forEach((text,i)=>{
    const rk = respKey(r,vi,i), open = openResps.has(rk);
    const g = (v.groups||[])[i];
    const snip = (text||'').replace(/\s+/g,' ').slice(0,150);
    h += '<div class="fresp" style="border-left-color:'+gcolor(g)+'">'
       + '<div class="fresp-head" data-resp="'+esc(rk)+'">'
       + '<span class="badge">#'+(i+1)+(g!=null&&g>=0?(' · g'+g):'')+'</span>'
       + (open ? '' : '<span class="snip">'+esc(snip)+'</span>')
       + '</div>'
       + (open ? '<div class="full">'+esc(text)+'</div>' : '')
       + '</div>';
  });
  h += '</div></div>';
  return h;
}

function contingencyHtml(r){
  const j = r.judge||{};
  if (!j.contingency || !j.contingency.length) return '';
  const groups = j.group_ids || j.contingency[0].map((_,i)=>i);
  let head = '<tr><th>variant \\ judge group</th>'+groups.map(g=>'<th><span class="box" style="background:'+gcolor(g)+'"></span> g'+g+'</th>').join('')+'</tr>';
  let rows = '';
  r.variants.forEach((v,vi)=>{
    const cells = (j.contingency[vi]||[]).map(c=>'<td>'+(c||'')+'</td>').join('');
    rows += '<tr><td class="v">'+esc(v.variant)+'</td>'+cells+'</tr>';
  });
  return '<table class="cont">'+head+rows+'</table>';
}

function render(){
  const rows = sortRecords(FAM.records.filter(passes));
  $('#cards').innerHTML = rows.length
    ? rows.map(renderCard).join('')
    : '<div class="empty">No framing-family bundles match the current filters.</div>';
  // dashboard over the filtered set
  const nc = rows.filter(r=>(r.judge||{}).contradiction).length;
  const aris = rows.map(r=>(r.judge||{}).ari).filter(x=>x!=null);
  const ma = aris.length ? mean(aris) : null;
  const sws = rows.map(r=>r.swing).filter(x=>x!=null);
  const fams = new Set(rows.map(r=>r.family));
  $('#dash').innerHTML =
      '<span class="chip"><b>'+rows.length+'</b> bundles</span>'
    + '<span class="chip"><b>'+fams.size+'</b> families</span>'
    + '<span class="chip hot"><b>'+nc+'</b> contradictions</span>'
    + '<span class="chip">mean judge ARI <b>'+fmt(ma)+'</b></span>'
    + '<span class="chip">max swing <b>'+fmt(sws.length?Math.max.apply(null,sws):null)+'</b></span>';
}

// ---- grouped-bar chart: x = family; bars are either per-model or per-cohort ----
// "slots" are the fixed bar positions within each family band (so the same model /
// cohort lands in the same slot across families).
function slotList(){
  const sel = selModels();
  if (STATE.cmode==='cohort' && hasCohorts())
    return COHORTS.filter(ch=> FAM.models.some(m=> cohortOf(m)===ch && sel.has(m)))
      .map(ch=>({key:ch, label:COHORT_LABEL[ch], color:COHORT_COLOR[ch]}));
  return FAM.models.filter(m=> sel.has(m)).map(m=>({key:m, label:m, color:mcolor(m)}));
}

// bars for one family. cohort mode -> mean ± SD across that cohort's selected
// models; model mode -> one bar per selected model.
function chartBars(fid, metric){
  const sel = selModels();
  if (STATE.cmode==='cohort' && hasCohorts()){
    const out = [];
    for (const ch of COHORTS){
      const vals = [];
      for (const m of FAM.models){
        if (!sel.has(m) || cohortOf(m)!==ch) continue;
        const rec = FAM.records.find(r=> r.model===m && r.family===fid);
        const v = rec ? rec.metrics[metric] : null;
        if (v!=null) vals.push(v);
      }
      if (vals.length) out.push({key:ch, color:COHORT_COLOR[ch], val:mean(vals),
        std:vals.length>1?pstd(vals):0, n:vals.length, cohort:ch});
    }
    return out;
  }
  const out = [];
  for (const m of FAM.models){
    if (!sel.has(m)) continue;
    const rec = FAM.records.find(r=> r.model===m && r.family===fid);
    const v = rec ? rec.metrics[metric] : null;
    if (v==null) continue;
    out.push({key:m, color:mcolor(m), val:v, std:0, n:1, model:m});
  }
  return out;
}

function chartGroups(){
  const metric = STATE.metric, fids = Object.keys(FAM.families).sort(), out = [];
  for (const fid of fids){
    const bars = chartBars(fid, metric);
    if (bars.length) out.push({fid, label:fid, bars});
  }
  out.sort((a,b)=> mean(b.bars.map(x=>x.val)) - mean(a.bars.map(x=>x.val)));  // most framing-driven first
  return out;
}

function drawChart(){
  const root = $('#chartsvg'); root.innerHTML = '';
  const groups = chartGroups(), slots = slotList();
  if (!groups.length || !slots.length){
    root.appendChild(elh('div', {class:'cc-cap'}, 'Nothing to plot — pick at least one model.'));
    return;
  }
  const slotIdx = {}; slots.forEach((s2,i)=> slotIdx[s2.key]=i);
  const nb = slots.length, G = groups.length;
  const band = Math.max(46, (STATE.cmode==='cohort'?40:16)*nb + 18);
  const PAD = {l:48,r:14,t:14,b:108};
  const W = PAD.l + PAD.r + G*band, H = 340;
  let maxY = 0; groups.forEach(g=> g.bars.forEach(b=> maxY=Math.max(maxY, b.val+(b.std||0))));
  maxY = Math.max(maxY, 0.0001); if (maxY < 1) maxY = Math.min(1, maxY*1.15);
  const plotH = H - PAD.t - PAD.b, y = (v)=> PAD.t + plotH*(1 - v/maxY);
  const s = svgEl('svg', {width:W, height:H, viewBox:'0 0 '+W+' '+H});
  const NT = 5;
  for (let i=0;i<=NT;i++){
    const v=maxY*i/NT, yy=y(v);
    s.appendChild(svgEl('line', {x1:PAD.l, y1:yy, x2:W-PAD.r, y2:yy, class:i?'grid':'axis'}));
    s.appendChild(svgEl('text', {x:PAD.l-7, y:yy+4, 'text-anchor':'end', 'font-size':10}, v.toFixed(2)));
  }
  s.appendChild(svgEl('text', {x:13, y:PAD.t+plotH/2, 'text-anchor':'middle', 'font-size':11,
    transform:'rotate(-90 13 '+(PAD.t+plotH/2)+')'}, mlabel(STATE.metric)));
  groups.forEach((g, gi)=>{
    const x0 = PAD.l + gi*band, bw = Math.min((band-14)/nb, STATE.cmode==='cohort'?44:30);
    const start = x0 + (band - bw*nb)/2;
    g.bars.forEach((b)=>{
      const j = slotIdx[b.key]; if (j==null) return;
      const bx = start + j*bw, by = y(b.val), bh = (PAD.t+plotH) - by;
      const rect = svgEl('rect', {x:bx+1, y:by, width:Math.max(1,bw-2), height:Math.max(0,bh),
        fill:b.color, rx:2, class:'bar'});
      const tip = (b.cohort ? COHORT_LABEL[b.cohort] : b.model) + ' · ' + g.label + '\n'
        + mlabel(STATE.metric) + '=' + fmt(b.val)
        + (b.std>0 ? (' ± '+fmt(b.std)+' SD') : '')
        + (b.cohort ? ('\n'+b.n+' model'+(b.n===1?'':'s')+' (mean across cohort) — click for per-model')
                    : '\n(click to open this bundle)');
      rect.appendChild(svgEl('title', {}, tip));
      rect.addEventListener('click', ()=> b.cohort ? (STATE.cmode='model', saveState(), chartControls(), drawChart())
                                                   : focusCard(b.model, g.fid));
      s.appendChild(rect);
      if (b.std>0){  // ±1 SD across the cohort's models
        const cx=bx+bw/2, yhi=y(b.val+b.std), ylo=y(Math.max(0,b.val-b.std)), cap=Math.min(5,bw/3), ec='#e7ebf3';
        s.appendChild(svgEl('line', {x1:cx, y1:yhi, x2:cx, y2:ylo, stroke:ec, 'stroke-width':1.3}));
        s.appendChild(svgEl('line', {x1:cx-cap, y1:yhi, x2:cx+cap, y2:yhi, stroke:ec, 'stroke-width':1.3}));
        s.appendChild(svgEl('line', {x1:cx-cap, y1:ylo, x2:cx+cap, y2:ylo, stroke:ec, 'stroke-width':1.3}));
      }
    });
    const lx = x0 + band/2, ly = PAD.t+plotH+12;
    const t = svgEl('text', {x:lx, y:ly, 'text-anchor':'end', 'font-size':10, class:'xlbl',
      transform:'rotate(-35 '+lx+' '+ly+')'}, g.label.length>18?g.label.slice(0,17)+'…':g.label);
    t.appendChild(svgEl('title', {}, (FAM.families[g.fid]||{}).title || g.label));
    s.appendChild(t);
  });
  s.appendChild(svgEl('line', {x1:PAD.l, y1:PAD.t+plotH, x2:W-PAD.r, y2:PAD.t+plotH, class:'axis'}));
  const wrap = elh('div', {class:'cc-scroll'}); wrap.appendChild(s);
  root.appendChild(wrap);
}

function chartControls(){
  const bar = $('#chartctl'); bar.innerHTML='';
  const chip = (label, on, sw, onclick)=>{ const c=elh('button',{class:'cc-chip'+(on?' on':'')});
    if(sw) c.appendChild(elh('span',{class:'sw',style:'background:'+sw})); c.appendChild(document.createTextNode(label));
    c.addEventListener('click', onclick); return c; };
  // view mode: by cohort (fine-tuned vs base) | by model
  if (hasCohorts()){
    const vGrp = elh('div', {class:'cc-grp'});
    vGrp.appendChild(elh('span', {class:'cc-lbl'}, 'view'));
    [['cohort','by cohort'],['model','by model']].forEach(([k,lab])=>{
      vGrp.appendChild(chip(lab, STATE.cmode===k, null, ()=>{ STATE.cmode=k; saveState(); chartControls(); drawChart(); chartCaption(); }));
    });
    bar.appendChild(vGrp);
  }
  // metric
  const mGrp = elh('div', {class:'cc-grp'});
  mGrp.appendChild(elh('span', {class:'cc-lbl'}, 'metric'));
  const sel = elh('select');
  METRICS.forEach(m=>{ const o=elh('option', {value:m.key}, m.label); if(m.key===STATE.metric)o.setAttribute('selected',''); sel.appendChild(o); });
  sel.value = STATE.metric;
  sel.addEventListener('change', ()=>{ STATE.metric=sel.value; saveState(); drawChart(); chartCaption(); });
  mGrp.appendChild(sel); bar.appendChild(mGrp);
  // cohort colour legend (only meaningful in cohort view)
  if (hasCohorts() && STATE.cmode==='cohort'){
    const lGrp = elh('div', {class:'cc-grp'});
    COHORTS.forEach(ch=>{ const sp=elh('span',{class:'cc-lbl'});
      sp.appendChild(elh('span',{class:'sw',style:'background:'+COHORT_COLOR[ch]+';display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px'}));
      sp.appendChild(document.createTextNode(COHORT_LABEL[ch])); lGrp.appendChild(sp); });
    bar.appendChild(lGrp);
  }
  // model membership toggles (also drive which models go into each cohort mean)
  const cur = selModels();
  const moGrp = elh('div', {class:'cc-grp'});
  moGrp.appendChild(elh('span', {class:'cc-lbl'}, 'models'));
  moGrp.appendChild(chip('all', false, null, ()=>{ setSelModels(new Set(FAM.models)); saveState(); chartControls(); drawChart(); }));
  moGrp.appendChild(chip('none', false, null, ()=>{ setSelModels(new Set()); saveState(); chartControls(); drawChart(); }));
  if (hasCohorts()) COHORTS.forEach(ch=> moGrp.appendChild(chip('only '+ch, false, COHORT_COLOR[ch], ()=>{
    setSelModels(new Set(FAM.models.filter(m=>cohortOf(m)===ch))); saveState(); chartControls(); drawChart(); })));
  FAM.models.forEach(m=> moGrp.appendChild(chip(m, cur.has(m), hasCohorts()?COHORT_COLOR[cohortOf(m)]:mcolor(m), ()=>{
    const s2 = selModels(); s2.has(m)?s2.delete(m):s2.add(m); setSelModels(s2); saveState(); chartControls(); drawChart(); })));
  bar.appendChild(moGrp);
}
function chartCaption(){
  const base = 'Cross-variant metric over the pooled judge / embeddings — the framing-split signal '
    + '(higher = answer more framing-driven).';
  $('#chartcap').textContent = base + (STATE.cmode==='cohort' && hasCohorts()
    ? ' Two bars per family: mean across each cohort, error bars ±1 SD over its models. Click a cohort bar for the per-model breakdown.'
    : ' One bar per model. Click a bar to open that bundle.');
}

// jump from a chart bar to a bundle card: open it, highlight, scroll into view.
function focusCard(model, family){
  const key = model + NUL + family;
  // make sure it passes the filters so it is actually in the DOM
  if (STATE.model!=='__all__' && STATE.model!==model) STATE.model='__all__';
  if (STATE.family!=='__all__' && STATE.family!==family) STATE.family='__all__';
  STATE.onlyContra = false;
  focusKey = key; openCards.add(key); saveState(); syncControls(); render();
  const el = document.querySelector('.card[data-key="'+CSS.escape(key)+'"]');
  (el || $('#cards')).scrollIntoView({behavior:'smooth', block:'start'});
}

// ---- controls wiring ----
function opts(sel, vals, withAll, allLabel){
  const el = $(sel); el.innerHTML='';
  if (withAll) el.append(new Option(allLabel||'(all)', '__all__'));
  vals.forEach(v=> el.append(new Option(v, v)));
}
function syncControls(){
  $('#model').value = STATE.model; $('#family').value = STATE.family;
  $('#sort').value = STATE.sort; $('#search').value = STATE.search;
  $('#onlyContra').checked = STATE.onlyContra;
}
function wire(){
  const bind = (id, k, ev, get)=> $(id).addEventListener(ev, ()=>{ STATE[k]=get(); focusKey=null; saveState(); render(); });
  bind('#model','model','change', ()=>$('#model').value);
  bind('#family','family','change', ()=>$('#family').value);
  bind('#sort','sort','change', ()=>$('#sort').value);
  bind('#search','search','input', ()=>$('#search').value.trim());
  bind('#onlyContra','onlyContra','change', ()=>$('#onlyContra').checked);
  $('#expandAll').addEventListener('click', ()=>{ sortRecords(FAM.records.filter(passes)).forEach(r=>openCards.add(cardKey(r))); render(); });
  $('#collapseAll').addEventListener('click', ()=>{ openCards.clear(); render(); });
  $('#reset').addEventListener('click', ()=>{ STATE=Object.assign({}, DEFAULTS); focusKey=null; openCards.clear(); openResps.clear();
    syncControls(); saveState(); chartControls(); drawChart(); render(); });
  $('#cards').addEventListener('click', (e)=>{
    const rh = e.target.closest('.fresp-head');
    if (rh){ const k=rh.dataset.resp; openResps.has(k)?openResps.delete(k):openResps.add(k); render(); return; }
    const ch = e.target.closest('.card-head');
    if (ch){ const k=ch.dataset.card; openCards.has(k)?openCards.delete(k):openCards.add(k); render(); }
  });
}

function init(){
  STATE = loadState();
  if (!hasCohorts()) STATE.cmode = 'model';  // need both cohorts to compare them
  const fids = Object.keys(FAM.families).sort();
  opts('#model', FAM.models.slice(), true);
  opts('#family', fids, true);
  if (!FAM.models.includes(STATE.model)) STATE.model='__all__';
  if (!fids.includes(STATE.family)) STATE.family='__all__';
  syncControls();
  // safety net: never open to a blank screen from stale shared filter state
  if (FAM.records.filter(passes).length === 0){ STATE=Object.assign({}, DEFAULTS); syncControls(); saveState(); }
  wire();
  chartControls(); drawChart(); chartCaption();
  // auto-expand the most framing-driven bundles on first load
  FAM.records.filter(r=> (r.judge||{}).contradiction || ((r.judge||{}).ari||0) >= 0.4)
    .forEach(r=> openCards.add(cardKey(r)));
  render();
}
init();
