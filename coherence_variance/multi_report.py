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

from pathlib import Path

from . import category_chart
from . import metrics as metrics_mod
from .report_ui import BASE_CSS, BASE_JS, html_document, json_blob


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


# Multi-report-specific styles on top of report_ui.BASE_CSS.
_CSS = """
select#view{border-color:var(--accent);font-weight:600;}
.pill{font-size:11px;padding:1px 8px;border-radius:10px;font-weight:600;}
.stab{width:46px;height:6px;border-radius:3px;background:#2a2f3a;overflow:hidden;flex:none;}
.stab i{display:block;height:100%;}
.drift{font-size:10px;color:var(--amber);font-weight:600;}
.matrix-tog{font-size:11px;color:var(--accent);cursor:pointer;margin:8px 0 2px;display:inline-block;}
.matrix{display:grid;gap:1px;margin:6px 0;}
.matrix .cell{width:13px;height:13px;border-radius:1px;}
.legend{font-size:11px;color:var(--muted);margin:4px 0;}
"""

_JS = r"""
// green(high)->amber->red(low)
const grade=(x)=> x>=0.85?'#5ad19a':(x>=0.7?'#ffd24a':'#ff6b6b');

let openCards=new Set(), openResps=new Set(), openHeat=new Set();
let qFilter=null;   // set by clicking a question bar in the category chart
const ck=(b)=>b.model+NUL+b.question_id;
const rk=(b,i)=>ck(b)+NUL+i;

const DEFAULTS={view:'consensus',model:'__all__',group:'__all__',sort:'consistency'};
const STORE=stateStore('multi_report_state_v1:'+(DATA.run_labels||[]).join('_'), DEFAULTS);
const saveSel=()=> STORE.save({view:$('#view').value, model:$('#model').value,
  group:$('#group').value, sort:$('#sort').value});

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
  opts('#model', DATA.models, ['(all)','__all__']);
  const groups=[...new Set(DATA.bundles.map(b=>b.group))].filter(Boolean);
  opts('#group', groups, ['(all)','__all__']);
  // restore persisted filter state (ignore values that don't fit this dataset)
  const st=STORE.load();
  $('#view').value = (st.view==='consensus'||DATA.run_labels.includes(st.view)) ? st.view : 'consensus';
  $('#model').value = (st.model==='__all__'||DATA.models.includes(st.model)) ? st.model : '__all__';
  $('#group').value = (st.group==='__all__'||groups.includes(st.group)) ? st.group : '__all__';
  $('#sort').value = ['consistency','groups','model','group'].includes(st.sort) ? st.sort : 'consistency';
  ['#view','#model','#group','#sort'].forEach(s=>$(s).addEventListener('change',()=>{ saveSel(); render(); }));
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
    data_json = json_blob(data)
    chart_data = json_blob(category_chart.build_chart_data_multi(runs))
    chart_section = category_chart.chart_section_html("cchart")
    o = agg["overall"]
    sub = (
        f"{o['n_runs']} judge runs ({', '.join(o['run_labels'])}) · {o['n_bundles']} bundles · "
        f"judge self-consistency view"
    )
    body = f"""<header>
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
<script>{BASE_JS}</script>
<script>const CHART = {chart_data};</script>
<script>{category_chart.CHART_JS}</script>
<script>const DATA = {data_json};</script>
<script>{_JS}</script>"""
    out_path.write_text(
        html_document(
            "Judge runs — multi-view",
            BASE_CSS + _CSS + category_chart.CHART_CSS,
            body,
        )
    )
    return out_path
