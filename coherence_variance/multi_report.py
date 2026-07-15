"""Unified multi-judge-run viewer.

One self-contained HTML that embeds every judge run of a generation plus the
cross-run consensus. A top "view" selector switches between each individual run
(its own groupings) and **consensus** (the aggregate):

- individual run: bundles grouped by that run's judge partition (+ entropy);
- consensus: bundles grouped by the majority-vote partition, every response
  shaded by placement *stability* (drifters flagged), headline consensus-strength
  / ARI / contested-pairs, and an on-demand co-association heatmap per bundle.

Progressive disclosure keeps it readable: one scalar per bundle by default,
which-responses-drift on expand, full agreement matrix only on click.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import category_chart
from . import metrics as metrics_mod


def build_view_data(runs: dict[str, dict], agg: dict) -> dict:
    run_labels = list(runs)
    idx = {
        lab: {(r["model"], r["question_id"]): r for r in a["results"]}
        for lab, a in runs.items()
    }
    qmeta = next(iter(runs.values())).get("questions", {}) if runs else {}
    cons_idx = {(b["model"], b["question_id"]): b for b in agg["per_bundle"]}

    bundles = []
    for key in sorted(cons_idx):
        model, qid = key
        recs = {lab: idx[lab][key] for lab in run_labels if key in idx[lab]}
        any_rec = next(iter(recs.values()))
        per_run = {}
        for lab, r in recs.items():
            jl = r.get("judge_labels") or []
            j = r.get("judge") or {}
            per_run[lab] = {
                "labels": jl,
                "n_groups": j.get("n_groups"),
                "contradiction": bool(j.get("contradiction")),
                "rationale": j.get("rationale", ""),
                "flags": j.get("flags") or [],
                "entropy": metrics_mod.group_entropy(jl),
            }
        cb = cons_idx[key]
        bundles.append(
            {
                "model": model,
                "question_id": qid,
                "group": cb.get("group", ""),
                "responses": any_rec["responses"],
                "per_run": per_run,
                "consensus": {
                    "labels": cb["consensus_labels"],
                    "stability": cb["consensus_stability"],
                    "strength": cb["consensus_strength"],
                    "contested_pairs": cb["contested_pairs"],
                    "n_drifters": cb["n_drifters"],
                    "mean_ari": cb["mean_pairwise_ari"],
                    "n_groups": cb["n_groups"],
                    "coassoc": cb["coassoc"],
                },
            }
        )
    return {
        "run_labels": run_labels,
        "models": sorted({b["model"] for b in bundles}),
        "questions": qmeta,
        "overall": agg["overall"],
        "per_model": agg["per_model"],
        "bundles": bundles,
    }


_CSS = """
:root{--bg:#0f1115;--card:#1a1d24;--card2:#13161c;--muted:#8b93a7;--fg:#e6e9ef;--line:#2a2f3a;
--accent:#4f9dff;--red:#ff6b6b;--amber:#ffd24a;--green:#5ad19a;}
*{box-sizing:border-box;} body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;}
header{padding:14px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;
background:rgba(15,17,21,.97);backdrop-filter:blur(6px);z-index:10;}
h1{font-size:18px;margin:0;} .dash{color:var(--muted);font-size:12.5px;margin-top:5px;}
.dash b{color:var(--fg);} .dash .chip{margin-right:14px;}
.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:12px;}
.controls label{font-size:11.5px;color:var(--muted);display:flex;gap:5px;align-items:center;}
select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 7px;font-size:12.5px;}
select#view{border-color:var(--accent);font-weight:600;}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;}
button:hover{border-color:var(--accent);}
#cards{padding:14px 20px;display:flex;flex-direction:column;gap:10px;max-width:1180px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;}
.card.open{border-color:#39404e;}
.card-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:11px 14px;cursor:pointer;user-select:none;}
.card-head:hover{background:#1e222b;} .chev{color:var(--muted);font-size:11px;width:10px;transition:transform .15s;}
.card.open .chev{transform:rotate(90deg);}
.tag{font-size:11px;padding:1px 7px;border-radius:10px;background:#2a2f3a;color:var(--muted);}
.tag.model{color:#cfe0ff;background:#22304a;} .tag.group{color:#d9ffe6;background:#1f3a2c;}
.stats{font-size:11.5px;color:var(--muted);display:flex;gap:12px;flex-wrap:wrap;margin-left:auto;align-items:center;}
.stats b{color:var(--fg);}
.pill{font-size:11px;padding:1px 8px;border-radius:10px;font-weight:600;}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.body{padding:0 14px 12px;border-top:1px solid var(--line);}
.q{color:var(--muted);font-size:12px;white-space:pre-wrap;margin:10px 0;max-height:130px;overflow:auto;border-left:2px solid var(--line);padding-left:9px;}
.rationale{font-size:12.5px;margin:8px 0;} .flags{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;}
.flag{font-size:11px;padding:1px 8px;border-radius:10px;background:#3a2a2a;color:#ffb3b3;}
.sep{font-size:11px;color:var(--muted);margin:10px 0 4px;display:flex;gap:8px;align-items:center;}
.sep .box{width:11px;height:11px;border-radius:3px;}
.resp{border-left:4px solid;background:var(--card2);border-radius:0 6px 6px 0;margin:5px 0;}
.resp-head{display:flex;gap:9px;align-items:center;padding:6px 10px;cursor:pointer;}
.resp-head:hover{background:#171b22;}
.resp .badge{font-size:10.5px;color:var(--muted);font-family:ui-monospace,monospace;white-space:nowrap;}
.resp .swatch{width:9px;height:9px;border-radius:2px;display:inline-block;}
.resp .snip{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;}
.resp .full{white-space:pre-wrap;font-size:12.5px;padding:2px 12px 10px;}
.stab{width:46px;height:6px;border-radius:3px;background:#2a2f3a;overflow:hidden;flex:none;}
.stab i{display:block;height:100%;}
.drift{font-size:10px;color:var(--amber);font-weight:600;}
.matrix-tog{font-size:11px;color:var(--accent);cursor:pointer;margin:8px 0 2px;display:inline-block;}
.matrix{display:grid;gap:1px;margin:6px 0;}
.matrix .cell{width:13px;height:13px;border-radius:1px;}
.legend{font-size:11px;color:var(--muted);margin:4px 0;}
.empty{color:var(--muted);padding:24px;text-align:center;}
.qfocus{margin:10px 20px 0;max-width:1180px;padding:7px 12px;border-radius:8px;background:#1f2a3a;
  border:1px solid #2f3f57;color:#cfe0ff;font-size:12px;display:flex;gap:10px;align-items:center;}
.qfocus b{color:#fff;} .qfocus .x{margin-left:auto;cursor:pointer;color:var(--muted);}
.qfocus .x:hover{color:var(--fg);}
"""

_JS = r"""
const $=(s)=>document.querySelector(s);
const PALETTE=['#4f9dff','#ff8a5c','#5ad19a','#c98bff','#ffd24a','#ff6b9d','#6be0e0','#b0b85a','#e0846b','#8a9bff','#7ad17a','#d99bff'];
const gcolor=(i)=> i<0?'#666':PALETTE[i%PALETTE.length];
const esc=(s)=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt=(x)=>(x==null||isNaN(x))?'–':Number(x).toFixed(2);
const pct=(x)=>(x==null)?'–':Math.round(x*100)+'%';
const entropyOf=(labels)=>{if(!labels||!labels.length)return 0;const c={};labels.forEach(l=>c[l]=(c[l]||0)+1);
  const n=labels.length;let h=0;for(const k in c){const p=c[k]/n;h-=p*Math.log(p);}return h;};
// green(high)->amber->red(low)
const grade=(x)=> x>=0.85?'#5ad19a':(x>=0.7?'#ffd24a':'#ff6b6b');

let openCards=new Set(), openResps=new Set(), openHeat=new Set();
let qFilter=null;   // set by clicking a question bar in the category chart
const ck=(b)=>b.model+'\x1f'+b.question_id;
const rk=(b,i)=>ck(b)+'\x1f'+i;

function opts(sel,vals,extra){const el=$(sel);el.innerHTML='';if(extra)el.append(new Option(extra[0],extra[1]));
  vals.forEach(v=>el.append(new Option(v,v)));}

function groupedOrder(labels){ // returns [[idx,label]...] sorted by label, with original index tiebreak
  const idx=labels.map((l,i)=>[i,l]); idx.sort((a,b)=>(a[1]-b[1])||(a[0]-b[0])); return idx;}

function renderResp(b,i,mode){
  const text=b.responses[i]||'', key=rk(b,i), open=openResps.has(key);
  const cs=b.consensus, snip=text.replace(/\s+/g,' ').slice(0,150);
  let badge, border;
  if(mode==='consensus'){
    const g=cs.labels[i], stab=cs.stability[i]??1;
    border=gcolor(g);
    const drift = stab<0.6 ? '<span class="drift">drift</span>' : '';
    badge='<span class="badge">#'+(i+1)+'</span>'
        +'<span class="swatch" style="background:'+gcolor(g)+'"></span>'
        +'<span class="stab" title="placement stability '+fmt(stab)+'"><i style="width:'+Math.round(stab*100)+'%;background:'+grade(stab)+'"></i></span>'
        +drift;
  } else {
    const g=(b.per_run[mode].labels||[])[i]??-1; border=gcolor(g);
    badge='<span class="badge">#'+(i+1)+'</span><span class="swatch" style="background:'+gcolor(g)+'"></span><span class="badge">g'+g+'</span>';
  }
  let h='<div class="resp" style="border-left-color:'+border+'"><div class="resp-head" data-resp="'+esc(key)+'">'+badge
      +(open?'':'<span class="snip">'+esc(snip)+'</span>')+'</div>';
  if(open) h+='<div class="full">'+esc(text)+'</div>';
  return h+'</div>';
}

function renderResponses(b,labels,mode){
  const order=groupedOrder(labels);
  const counts={}; order.forEach(([i,l])=>counts[l]=(counts[l]||0)+1);
  let h='', last;
  for(const [i,l] of order){
    if(l!==last){ last=l;
      const kind = mode==='consensus' ? 'Consensus group' : 'Group';
      h+='<div class="sep"><span class="box" style="background:'+gcolor(l)+'"></span>'+kind+' '+l+' — '+counts[l]+'</div>';
    }
    h+=renderResp(b,i,mode);
  }
  return h;
}

function renderMatrix(b){
  const cs=b.consensus, C=cs.coassoc, n=C.length;
  if(!n) return '';
  // order rows/cols by consensus group then stability (blocks become visible)
  const order=[...Array(n).keys()].sort((a,z)=> (cs.labels[a]-cs.labels[z]) || ((cs.stability[z]??1)-(cs.stability[a]??1)) || (a-z));
  let cells='';
  for(const a of order) for(const z of order){
    const v=C[a][z]; // 1 = always same group, 0 = never
    const L=12+v*72; // dark->light
    const contested = (v>0.25&&v<0.75) ? ';outline:1px solid #ffd24a55' : '';
    cells+='<div class="cell" title="#'+(a+1)+' & #'+(z+1)+': '+fmt(v)+' of runs same group" style="background:hsl(210,18%,'+L+'%)'+contested+'"></div>';
  }
  return '<div class="legend">agreement matrix — light = always grouped together, dark = never, amber outline = contested ('+n+'×'+n+', ordered by consensus group)</div>'
       +'<div class="matrix" style="grid-template-columns:repeat('+n+',13px)">'+cells+'</div>';
}

function renderCard(b){
  const view=$('#view').value, key=ck(b), isOpen=openCards.has(key);
  const cs=b.consensus;
  let head='<div class="card'+(isOpen?' open':'')+'"><div class="card-head" data-card="'+esc(key)+'">'
    +'<span class="chev">▶</span><span class="tag model">'+esc(b.model)+'</span>'
    +'<span class="tag group">'+esc(b.group)+'</span><span class="tag">'+esc(b.question_id)+'</span>';
  let stats='';
  if(view==='consensus'){
    stats='<span class="pill" style="background:'+grade(cs.strength)+'22;color:'+grade(cs.strength)+'">consensus '+fmt(cs.strength)+'</span>'
      +'<span>ARI=<b>'+fmt(cs.mean_ari)+'</b></span>'
      +'<span>n_groups/run=<b>'+cs.n_groups.join('/')+'</b></span>'
      +'<span>contested pairs=<b>'+cs.contested_pairs+'</b></span>'
      +'<span>drifters=<b>'+cs.n_drifters+'</b></span>';
  } else {
    const pr=b.per_run[view];
    if(!pr){ return ''; }
    stats=(pr.contradiction?'<span class="dot" style="background:var(--red)" title="contradiction"></span>':'')
      +'<span>judge groups=<b>'+(pr.n_groups??'–')+'</b></span>'
      +'<span>H=<b>'+fmt(pr.entropy!=null?pr.entropy:entropyOf(pr.labels))+'</b></span>';
  }
  head+='<span class="stats">'+stats+'</span></div>';
  if(!isOpen) return head+'</div>';

  let body='<div class="body"><div class="q">'+esc((DATA.questions[b.question_id]||{}).prompt||b.question_id)+'</div>';
  if(view==='consensus'){
    body+='<div class="rationale">Consensus grouping across '+DATA.run_labels.length+' judge runs; each response shaded by how stably it is placed (drifters = the judge can\'t consistently group them).</div>';
    body+=renderResponses(b,cs.labels,'consensus');
    const ho=openHeat.has(key);
    body+='<div class="matrix-tog" data-heat="'+esc(key)+'">'+(ho?'▾ hide':'▸ show')+' agreement matrix</div>';
    if(ho) body+=renderMatrix(b);
  } else {
    const pr=b.per_run[view];
    if(pr.rationale) body+='<div class="rationale">'+esc(pr.rationale)+'</div>';
    if(pr.flags&&pr.flags.length) body+='<div class="flags">'+pr.flags.map(f=>'<span class="flag">'+esc(f)+'</span>').join('')+'</div>';
    body+=renderResponses(b,pr.labels,view);
  }
  return head+body+'</div></div>';
}

function sortBundles(rows){
  const view=$('#view').value, by=$('#sort').value;
  const ng=(b)=> view==='consensus' ? (b.consensus.n_groups.reduce((s,x)=>s+x,0)/b.consensus.n_groups.length) : ((b.per_run[view]||{}).n_groups||0);
  const cmps={
    consistency:(a,b)=> a.consensus.strength-b.consensus.strength || a.consensus.mean_ari-b.consensus.mean_ari, // least consistent first
    groups:(a,b)=> ng(b)-ng(a),
    model:(a,b)=> a.model.localeCompare(b.model)||a.group.localeCompare(b.group)||a.question_id.localeCompare(b.question_id),
    group:(a,b)=> a.group.localeCompare(b.group)||a.model.localeCompare(b.model)||a.question_id.localeCompare(b.question_id),
  };
  return rows.slice().sort(cmps[by]||cmps.consistency);
}

function render(){
  const view=$('#view').value, model=$('#model').value, group=$('#group').value;
  if(qFilter){
    const qp=(DATA.questions[qFilter]||{}).prompt||qFilter;
    $('#qfocus').style.display='flex';
    $('#qfocus').innerHTML='Showing bundles for question <b>'+esc(qFilter)+'</b> — '
      +esc(qp.length>120?qp.slice(0,119)+'…':qp)+'<span class="x" id="qfocusClear" title="show all questions">✕ clear</span>';
  } else { $('#qfocus').style.display='none'; $('#qfocus').innerHTML=''; }
  let rows=DATA.bundles.filter(b=> (model==='__all__'||b.model===model) && (group==='__all__'||b.group===group)
    && (!qFilter||b.question_id===qFilter));
  if(view!=='consensus') rows=rows.filter(b=>b.per_run[view]);
  rows=sortBundles(rows);
  $('#cards').innerHTML = rows.length ? rows.map(renderCard).join('') : '<div class="empty">No bundles match.</div>';
  const o=DATA.overall;
  $('#dash').innerHTML='<span class="chip">view: <b>'+(view==='consensus'?'consensus ('+DATA.run_labels.length+' runs)':view)+'</b></span>'
    +'<span class="chip"><b>'+rows.length+'</b> bundles</span>'
    +'<span class="chip">mean consensus <b>'+fmt(o.mean_consensus_strength)+'</b></span>'
    +'<span class="chip">mean ARI <b>'+fmt(o.mean_partition_ari)+'</b></span>'
    +'<span class="chip">contradiction unstable <b>'+pct(o.frac_contradiction_unstable)+'</b></span>';
}

// Jump from a chart bar to that question's bundles: filter, expand, scroll into view.
function focusQuestion(info){
  if(!info||!info.qid) return;
  qFilter=info.qid;
  DATA.bundles.forEach(b=>{ if(b.question_id===info.qid && (!info.model||b.model===info.model)) openCards.add(ck(b)); });
  render();
  const want=info.model?(info.model+'\x1f'+info.qid):null; let target=null;
  document.querySelectorAll('#cards .card-head').forEach(h=>{
    if(target) return;
    if(want?h.dataset.card===want:(h.dataset.card||'').endsWith('\x1f'+info.qid)) target=h;
  });
  (target||$('#cards')).scrollIntoView({behavior:'smooth',block:'start'});
}

function init(){
  opts('#view', DATA.run_labels, ['consensus','consensus']);
  $('#view').value='consensus';
  opts('#model', DATA.models, ['(all)','__all__']);
  opts('#group', [...new Set(DATA.bundles.map(b=>b.group))].filter(Boolean), ['(all)','__all__']);
  ['#view','#model','#group','#sort'].forEach(s=>$(s).addEventListener('change',render));
  $('#expandAll').addEventListener('click',()=>{DATA.bundles.forEach(b=>openCards.add(ck(b)));render();});
  $('#collapseAll').addEventListener('click',()=>{openCards.clear();render();});
  $('#qfocus').addEventListener('click',(e)=>{ if(e.target.id==='qfocusClear'){ qFilter=null; render(); }});
  if(window.initCategoryChart) initCategoryChart(CHART, 'cchart', {storageKey:'multi_'+(DATA.run_labels||[]).join('_'), onBarClick:focusQuestion});
  $('#cards').addEventListener('click',(e)=>{
    const rh=e.target.closest('.resp-head'); if(rh){const k=rh.dataset.resp;openResps.has(k)?openResps.delete(k):openResps.add(k);render();return;}
    const mt=e.target.closest('[data-heat]'); if(mt){const k=mt.dataset.heat;openHeat.has(k)?openHeat.delete(k):openHeat.add(k);render();return;}
    const ch=e.target.closest('.card-head'); if(ch){const k=ch.dataset.card;openCards.has(k)?openCards.delete(k):openCards.add(k);render();}
  });
  render();
}
init();
"""


def build_multi_report(runs: dict[str, dict], agg: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    data = build_view_data(runs, agg)
    data_json = json.dumps(data).replace("</", "<\\/")
    chart_data = json.dumps(category_chart.build_chart_data_multi(runs)).replace(
        "</", "<\\/"
    )
    chart_section = category_chart.chart_section_html("cchart")
    o = agg["overall"]
    sub = (
        f"{o['n_runs']} judge runs ({', '.join(o['run_labels'])}) · {o['n_bundles']} bundles · "
        f"judge self-consistency view"
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Judge runs — multi-view</title><style>{_CSS}{category_chart.CHART_CSS}</style></head><body>
<header>
  <h1>Judge runs &amp; consensus</h1>
  <div class="dash" id="dash"></div>
  <div class="dash" style="margin-top:2px">{sub}</div>
  <div class="controls">
    <label>view <select id="view"></select></label>
    <label>model <select id="model"></select></label>
    <label>group <select id="group"></select></label>
    <label>sort
      <select id="sort">
        <option value="consistency">least consistent</option>
        <option value="groups">most groups</option>
        <option value="model">model</option>
        <option value="group">group</option>
      </select></label>
    <button id="expandAll">expand all</button>
    <button id="collapseAll">collapse all</button>
  </div>
</header>
{chart_section}
<div id="qfocus" class="qfocus" style="display:none"></div>
<div id="cards"></div>
<script>const CHART = {chart_data};</script>
<script>{category_chart.CHART_JS}</script>
<script>const DATA = {data_json};</script>
<script>{_JS}</script>
</body></html>
"""
    out_path.write_text(html)
    return out_path
