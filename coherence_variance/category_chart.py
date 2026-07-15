"""Interactive grouped-bar chart for per-category / per-question response variance.

This is the **client-rendered** counterpart to ``category_bars.py`` (which owns the
static, paper-ready matplotlib PNG). Both reuse the same ``PALETTE`` /
``short_labels`` / ``METRICS`` / ``OVERALL_EXCLUDE`` from ``category_bars`` so the
interactive chart and the embedded PNG never disagree on colour, labels, or the
macro-average exclusion set.

One renderer (``CHART_JS`` → ``window.initCategoryChart(data, mountId, opts)``)
serves two hosts:

* ``report.py`` — a single judge pass; one value per (model, question) bar, no
  error bars. Built by :func:`build_chart_data`.
* ``multi_report.py`` — K repeated judge passes; judge-derived metrics carry a
  per-pass list, so the chart draws ±1 SD error bars across passes. Built by
  :func:`build_chart_data_multi`. (Embedding metrics are judge-invariant — the
  embeddings are fixed across passes — so they stay single-valued, no error bars.)

The chart has two modes:

* **aggregate** — x = a *selectable subset* of categories, bar per model, height =
  metric averaged over the category's questions (parity with the static PNG);
* **drill-down** — x = each individual question in one category, bar per model.
  Clicking a question bar fires ``opts.onBarClick`` so the host can scroll the
  reader to that question's actual responses (category → questions → responses).

The host supplies the compact ``CHART`` data blob (numbers only, a few KB) so the
single renderer works against both reports despite their different ``DATA`` shapes.
"""

from __future__ import annotations

from coherence_variance import category_bars as cb

# Canonical metric order + display labels. The three judge/embedding entropy
# metrics come from category_bars.METRICS; cosine distance is chart-only.
METRIC_ORDER = [
    "group_entropy",
    "n_judge_groups",
    "cluster_entropy",
    "mean_pairwise_cosine_dist",
]
METRIC_LABELS = {
    **cb.METRICS,
    "mean_pairwise_cosine_dist": "wording variety (embedding distance)",
}
# Only judge-derived metrics vary across repeated judge passes, so only these get
# error bars in the multi-run report; the embedding metrics are judge-invariant.
JUDGE_METRICS = ["group_entropy", "n_judge_groups"]
EMBED_METRICS = ["cluster_entropy", "mean_pairwise_cosine_dist"]


def _present_metrics(cells: list[dict]) -> list[str]:
    return [k for k in METRIC_ORDER if any(c["vals"].get(k) for c in cells)]


def _assemble(
    cells: list[dict],
    models_order: list[str],
    qmeta: dict,
    q_used: set,
    n_runs: int,
) -> dict:
    """Pack ``cells`` into the normalized CHART blob the JS renderer consumes."""
    present = {c["model"] for c in cells}
    models = [m for m in models_order if m in present]
    for c in cells:  # any model not in the declared order (defensive)
        if c["model"] not in models:
            models.append(c["model"])
    groups = sorted({c["group"] for c in cells})
    buckets = sorted({c.get("bucket") for c in cells if c.get("bucket")})
    metrics = _present_metrics(cells)
    return {
        "models": models,
        "model_labels": cb.short_labels(models) if models else {},
        "groups": groups,
        "buckets": buckets,
        "metrics": metrics,
        "metric_labels": {k: METRIC_LABELS.get(k, k) for k in metrics},
        "judge_metrics": [m for m in JUDGE_METRICS if m in metrics],
        "n_runs": n_runs,
        "overall_exclude": list(cb.OVERALL_EXCLUDE),
        "questions": {
            qid: {"prompt": (qmeta.get(qid) or {}).get("prompt", qid)} for qid in q_used
        },
        "cells": cells,
    }


def _is_family_q(qmeta: dict, qid: str) -> bool:
    """A cross-variant framing-family question. Within-prompt resampling is the
    wrong metric for these (the signal is the cross-variant split, see
    ``families.py`` / ``families_report.html``), so they are excluded from the
    within-prompt category chart rather than shown as meaningless low-variance bars.
    """
    return bool((qmeta.get(qid) or {}).get("family"))


def build_chart_data(analysis: dict) -> dict:
    """Single judge pass → one rep per (model, question) cell (no error bars).

    Cross-variant framing-family questions are excluded (see :func:`_is_family_q`).
    """
    results = analysis.get("results", [])
    qmeta = analysis.get("questions") or {}
    models_order = list(analysis.get("models") or sorted({r["model"] for r in results}))
    cells: list[dict] = []
    q_used: set = set()
    for r in results:
        if _is_family_q(qmeta, r["question_id"]):
            continue
        m = r.get("metrics") or {}
        vals = {k: [float(m[k])] for k in METRIC_ORDER if m.get(k) is not None}
        if not vals:
            continue
        cells.append(
            {
                "model": r["model"],
                "group": r.get("group") or "?",
                "bucket": (qmeta.get(r["question_id"]) or {}).get("bucket") or "?",
                "qid": r["question_id"],
                "vals": vals,
            }
        )
        q_used.add(r["question_id"])
    return _assemble(cells, models_order, qmeta, q_used, 1)


def build_chart_data_multi(runs: dict[str, dict]) -> dict:
    """K judge passes → judge metrics carry a per-pass list (→ ±SD error bars).

    ``runs`` maps judge-run label → analysis dict (the same mapping
    ``multi_report.build_multi_report`` receives). Embedding metrics stay
    single-valued because they do not depend on the judge.
    """
    labels = list(runs)
    if not labels:
        return _assemble([], [], {}, set(), 0)
    first = runs[labels[0]]
    qmeta = first.get("questions") or {}
    models_order = list(first.get("models") or [])
    idx = {
        lab: {(r["model"], r["question_id"]): r for r in a.get("results", [])}
        for lab, a in runs.items()
    }
    keys = sorted(set().union(*[set(i.keys()) for i in idx.values()]))

    cells: list[dict] = []
    q_used: set = set()
    for mdl, qid in keys:
        if _is_family_q(qmeta, qid):  # exclude cross-variant framing families
            continue
        recs = [idx[lab].get((mdl, qid)) for lab in labels]
        recs = [r for r in recs if r is not None]
        if not recs:
            continue
        group = next((r.get("group") or "?" for r in recs), "?")
        vals: dict[str, list[float]] = {}
        for k in JUDGE_METRICS:  # full per-pass list → error bar
            seq = [
                float((r.get("metrics") or {})[k])
                for r in recs
                if (r.get("metrics") or {}).get(k) is not None
            ]
            if seq:
                vals[k] = seq
        for k in EMBED_METRICS:  # judge-invariant → single value
            for r in recs:
                v = (r.get("metrics") or {}).get(k)
                if v is not None:
                    vals[k] = [float(v)]
                    break
        if not vals:
            continue
        bucket = (qmeta.get(qid) or {}).get("bucket") or "?"
        cells.append(
            {"model": mdl, "group": group, "bucket": bucket, "qid": qid, "vals": vals}
        )
        q_used.add(qid)

    for mdl, _qid in keys:  # models seen only in later runs (defensive)
        if mdl not in models_order:
            models_order.append(mdl)
    return _assemble(cells, models_order, qmeta, q_used, len(labels))


def chart_section_html(mount_id: str = "cchart") -> str:
    """The single mount point; ``initCategoryChart`` injects controls + SVG."""
    return f'<section id="{mount_id}" class="cchart"></section>'


CHART_CSS = """
.cchart { padding:10px 20px 4px; max-width:1180px; }
.cchart .cc-bar { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:8px; }
.cchart .cc-bar > .cc-grp { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.cchart .cc-lbl { font-size:11px; color:var(--muted,#8b93a7); }
.cchart .cc-modebtn, .cchart .cc-chip, .cchart .cc-quick {
  background:var(--card,#1a1d24); color:var(--fg,#e6e9ef); border:1px solid var(--line,#2a2f3a);
  border-radius:6px; padding:3px 9px; font-size:11.5px; cursor:pointer; }
.cchart .cc-modebtn.on { border-color:var(--accent,#4f9dff); color:#cfe0ff; background:#22304a; }
.cchart .cc-chip { border-radius:11px; padding:2px 9px; opacity:.5; }
.cchart .cc-chip.on { opacity:1; border-color:var(--accent,#4f9dff); }
.cchart .cc-chip .sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; vertical-align:middle; }
.cchart select { background:var(--card,#1a1d24); color:var(--fg,#e6e9ef);
  border:1px solid var(--line,#2a2f3a); border-radius:6px; padding:3px 7px; font-size:12px; }
.cchart .cc-scroll { overflow-x:auto; border:1px solid var(--line,#2a2f3a); border-radius:10px; background:var(--card2,#13161c); }
.cchart svg { display:block; }
.cchart svg text { fill:var(--muted,#8b93a7); }
.cchart svg .axis { stroke:#3a4150; }
.cchart svg .grid { stroke:#222732; }
.cchart svg .bar { cursor:pointer; }
.cchart svg .bar:hover { opacity:.82; }
.cchart svg .xlbl { cursor:pointer; }
.cchart svg .xlbl:hover { fill:var(--accent,#4f9dff); }
.cchart .cc-cap { color:var(--muted,#8b93a7); font-size:11px; margin:5px 2px 0; }
.cchart .cc-cap b { color:var(--fg,#e6e9ef); }
"""

# Wrapped in an IIFE so its internal `const`s never collide with the host report's
# top-level script scope (which already declares $, PALETTE, esc, fmt, …).
CHART_JS = r"""
(function(){
  const PALETTE = ['#4f9dff','#ff8a5c','#5ad19a','#c98bff','#ffd24a','#ff6b9d','#6be0e0','#b0b85a','#e0846b','#8a9bff','#7ad17a','#d99bff'];
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svg = (t,a,txt)=>{ const e=document.createElementNS(SVGNS,t); for(const k in (a||{})) e.setAttribute(k,a[k]); if(txt!=null) e.textContent=txt; return e; };
  const elh = (t,a,txt)=>{ const e=document.createElement(t); for(const k in (a||{})) e.setAttribute(k,a[k]); if(txt!=null) e.textContent=txt; return e; };
  const mean = (xs)=> xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;
  const pstd = (xs)=>{ if(xs.length<2) return 0; const m=mean(xs); return Math.sqrt(xs.reduce((s,x)=>s+(x-m)*(x-m),0)/xs.length); };
  const fmt = (x)=> (x==null||isNaN(x)) ? '–' : Number(x).toFixed(2);
  const trunc = (s,n)=> (s&&s.length>n) ? s.slice(0,n-1)+'…' : (s||'');

  window.initCategoryChart = function(DATA, mountId, opts){
    opts = opts || {};
    const root = document.getElementById(mountId);
    if (!root) return;
    if (!DATA || !DATA.cells || !DATA.cells.length || !DATA.metrics.length){ root.innerHTML=''; return; }

    const cidx = {}; DATA.models.forEach((m,i)=> cidx[m]=i);
    const color = (m)=> PALETTE[(cidx[m]||0) % PALETTE.length];
    const mlabel = (m)=> (DATA.model_labels && DATA.model_labels[m]) || m;
    const exclude = new Set(DATA.overall_exclude || []);
    const isJudge = (metric)=> (DATA.judge_metrics||[]).includes(metric);

    const SKEY = opts.storageKey ? (opts.storageKey + ':chart') : null;
    function load(){ if(!SKEY) return {}; try { return JSON.parse(localStorage.getItem(SKEY)||'{}')||{}; } catch(e){ return {}; } }
    function save(){ if(!SKEY) return; try { localStorage.setItem(SKEY, JSON.stringify(
      {mode:S.mode, metric:S.metric, cats:[...S.cats], buckets:[...S.buckets], models:[...S.models], drill:S.drill})); } catch(e){} }
    const sv = load();
    const BUCKETS = DATA.buckets || [];
    const hasBuckets = BUCKETS.length > 0;
    const validMode = (m)=> m==='drill' || (m==='bucket' && hasBuckets) || m==='agg';
    const S = {
      mode: validMode(sv.mode) ? sv.mode : 'agg',
      metric: DATA.metrics.includes(sv.metric) ? sv.metric : DATA.metrics[0],
      cats: new Set((sv.cats||[]).filter(c=>DATA.groups.includes(c))),
      buckets: new Set((sv.buckets||[]).filter(b=>BUCKETS.includes(b))),
      models: new Set((sv.models||[]).filter(m=>DATA.models.includes(m))),
      drill: DATA.groups.includes(sv.drill) ? sv.drill : DATA.groups[0],
    };
    if (!S.cats.size) DATA.groups.forEach(c=>S.cats.add(c));      // default: all categories
    if (!S.buckets.size) BUCKETS.forEach(b=>S.buckets.add(b));    // default: all buckets
    if (!S.models.size) DATA.models.forEach(m=>S.models.add(m));  // default: all models
    // bucket -> set of its categories (so a bucket bar drills into its categories)
    const b2g = {}; DATA.cells.forEach(c=>{ (b2g[c.bucket] || (b2g[c.bucket]=new Set())).add(c.group); });

    const cellVal = (c,metric)=>{ const v=c.vals[metric]; return (v&&v.length) ? v : null; };

    // aggregate mode: one group per selected category; bar height = metric averaged
    // over the category's questions; error bar = SD across judge passes of the
    // per-pass category mean (judge metrics, multi-run only).
    function aggBy(keyField, list, sel){
      const metric=S.metric, byCM={};
      for (const c of DATA.cells){
        const key=c[keyField];
        if (!sel.has(key) || !S.models.has(c.model)) continue;
        const reps=cellVal(c,metric); if(!reps) continue;
        (byCM[key] || (byCM[key]={}));
        (byCM[key][c.model] || (byCM[key][c.model]=[])).push(reps);
      }
      const out=[];
      for (const cat of list){
        if (!sel.has(cat)) continue;
        const mm=byCM[cat]||{}, bars=[];
        for (const model of DATA.models){
          if (!S.models.has(model) || !mm[model]) continue;
          const qreps=mm[model], K=Math.max.apply(null, qreps.map(r=>r.length));
          const perRep=[];
          for (let r=0;r<K;r++){
            const vs=qreps.map(rr=> rr.length>r ? rr[r] : (rr.length===1?rr[0]:null)).filter(v=>v!=null);
            if (vs.length) perRep.push(mean(vs));
          }
          bars.push({model, mean:mean(perRep), std:perRep.length>1?pstd(perRep):0, nq:qreps.length, k:K});
        }
        out.push({key:cat, label:cat, bars});
      }
      out.sort((a,b)=> (b.bars.length?mean(b.bars.map(x=>x.mean)):-1e9) - (a.bars.length?mean(a.bars.map(x=>x.mean)):-1e9));
      return out;
    }

    // drill mode: one group per question in the chosen category; error bar = SD
    // across judge passes for that single (model, question) bundle.
    function drillGroups(){
      const metric=S.metric, cat=S.drill, byQM={}, qids=new Set();
      for (const c of DATA.cells){
        if (c.group!==cat || !S.models.has(c.model)) continue;
        const reps=cellVal(c,metric); if(!reps) continue;
        (byQM[c.qid] || (byQM[c.qid]={}))[c.model]=reps; qids.add(c.qid);
      }
      const out=[];
      for (const qid of [...qids].sort()){
        const mm=byQM[qid]||{}, bars=[];
        for (const model of DATA.models){
          if (!S.models.has(model) || !mm[model]) continue;
          const reps=mm[model];
          bars.push({model, mean:mean(reps), std:reps.length>1?pstd(reps):0, nq:1, k:reps.length});
        }
        out.push({key:qid, label:qid, qid, bars});
      }
      return out;
    }

    function drawChart(groups){
      const selModels = DATA.models.filter(m=>S.models.has(m));
      const nb = Math.max(1, selModels.length), G = groups.length;
      const band = Math.max(46, 26*nb + 16);
      const PAD={l:54,r:14,t:14,b:104};
      const W = PAD.l + PAD.r + Math.max(1,G)*band, H = 360;
      let maxY = 0;
      groups.forEach(g=> g.bars.forEach(b=> { maxY = Math.max(maxY, b.mean + b.std); }));
      if (!(maxY>0)) maxY = 1;
      const plotH = H - PAD.t - PAD.b;
      const y = (v)=> PAD.t + plotH*(1 - v/maxY);
      const s = svg('svg', {width:W, height:H, viewBox:'0 0 '+W+' '+H});

      // y grid + ticks
      const NT=5;
      for (let i=0;i<=NT;i++){
        const v=maxY*i/NT, yy=y(v);
        s.appendChild(svg('line', {x1:PAD.l, y1:yy, x2:W-PAD.r, y2:yy, class:i?'grid':'axis'}));
        s.appendChild(svg('text', {x:PAD.l-7, y:yy+4, 'text-anchor':'end', 'font-size':10}, v.toFixed(2)));
      }
      // y label
      const yl = svg('text', {x:14, y:PAD.t+plotH/2, 'text-anchor':'middle', 'font-size':11,
        transform:'rotate(-90 14 '+(PAD.t+plotH/2)+')'}, DATA.metric_labels[S.metric]||S.metric);
      s.appendChild(yl);

      groups.forEach((g, gi)=>{
        const x0 = PAD.l + gi*band, inner = band - 14, bw = Math.min(inner/nb, 40);
        const start = x0 + (band - bw*nb)/2;
        g.bars.forEach((b)=>{
          const j = selModels.indexOf(b.model);
          const bx = start + j*bw, by = y(b.mean), bh = (PAD.t+plotH) - by;
          const rect = svg('rect', {x:bx+1, y:by, width:Math.max(1,bw-2), height:Math.max(0,bh),
            fill:color(b.model), rx:2, class:'bar'});
          const tip = mlabel(b.model)+' · '+g.label+'\n'+(DATA.metric_labels[S.metric]||S.metric)+'='+fmt(b.mean)
            + (b.std>0 ? (' ± '+fmt(b.std)+' (SD over '+b.k+' passes)') : '')
            + (S.mode==='agg' ? ('\n'+b.nq+' question'+(b.nq===1?'':'s')) : '');
          rect.appendChild(svg('title', {}, tip));
          rect.addEventListener('click', ()=> onBar(g, b));
          s.appendChild(rect);
          if (b.std>0){  // error bar: stem + caps at mean ± SD
            const cx=bx+bw/2, yhi=y(b.mean+b.std), ylo=y(Math.max(0,b.mean-b.std)), cap=Math.min(5,bw/3);
            const ec='#e7ebf3';
            s.appendChild(svg('line', {x1:cx, y1:yhi, x2:cx, y2:ylo, stroke:ec, 'stroke-width':1.3}));
            s.appendChild(svg('line', {x1:cx-cap, y1:yhi, x2:cx+cap, y2:yhi, stroke:ec, 'stroke-width':1.3}));
            s.appendChild(svg('line', {x1:cx-cap, y1:ylo, x2:cx+cap, y2:ylo, stroke:ec, 'stroke-width':1.3}));
          }
        });
        // x label (rotated), clickable
        const lx = x0 + band/2, ly = PAD.t+plotH+12;
        const t = svg('text', {x:lx, y:ly, 'text-anchor':'end', 'font-size':10, class:'xlbl',
          transform:'rotate(-35 '+lx+' '+ly+')'}, trunc(g.label, 16));
        const full = S.mode==='drill' ? ((DATA.questions[g.qid]||{}).prompt || g.label) : g.label;
        t.appendChild(svg('title', {}, full + (S.mode==='agg' ? ' — click to drill into questions' : ' — click for responses')));
        t.addEventListener('click', ()=> onLabel(g));
        s.appendChild(t);
      });
      // baseline
      s.appendChild(svg('line', {x1:PAD.l, y1:PAD.t+plotH, x2:W-PAD.r, y2:PAD.t+plotH, class:'axis'}));
      const wrap = elh('div', {class:'cc-scroll'}); wrap.appendChild(s);
      return wrap;
    }

    function onBar(g, b){
      if (S.mode==='bucket'){ S.mode='agg'; S.cats=new Set(b2g[g.key]||[]); save(); draw(); return; }
      if (S.mode==='agg'){ S.mode='drill'; S.drill=g.key; save(); draw(); return; }
      if (opts.onBarClick) opts.onBarClick({mode:'drill', group:S.drill, qid:g.qid, model:b.model});
    }
    function onLabel(g){
      if (S.mode==='bucket'){ S.mode='agg'; S.cats=new Set(b2g[g.key]||[]); save(); draw(); return; }
      if (S.mode==='agg'){ S.mode='drill'; S.drill=g.key; save(); draw(); return; }
      if (opts.onBarClick) opts.onBarClick({mode:'drill', group:S.drill, qid:g.qid, model:null});
    }

    function chip(label, on, sw, onclick){
      const c = elh('button', {class:'cc-chip'+(on?' on':'')});
      if (sw) c.appendChild(elh('span', {class:'sw', style:'background:'+sw}));
      c.appendChild(document.createTextNode(label));
      c.addEventListener('click', onclick);
      return c;
    }

    function controls(){
      const bar = elh('div', {class:'cc-bar'});
      // mode
      const modeGrp = elh('div', {class:'cc-grp'});
      modeGrp.appendChild(elh('span', {class:'cc-lbl'}, 'view'));
      const modes = [['agg','by category']];
      if (hasBuckets) modes.push(['bucket','by bucket']);
      modes.push(['drill','by question']);
      modes.forEach(([k,lab])=>{
        const b=elh('button', {class:'cc-modebtn'+(S.mode===k?' on':'')}, lab);
        b.addEventListener('click', ()=>{ S.mode=k; save(); draw(); });
        modeGrp.appendChild(b);
      });
      bar.appendChild(modeGrp);
      // metric
      const mGrp = elh('div', {class:'cc-grp'});
      mGrp.appendChild(elh('span', {class:'cc-lbl'}, 'metric'));
      const sel = elh('select');
      DATA.metrics.forEach(k=>{ const o=elh('option', {value:k}, DATA.metric_labels[k]||k); if(k===S.metric)o.setAttribute('selected','');
        sel.appendChild(o); });
      sel.value=S.metric;
      sel.addEventListener('change', ()=>{ S.metric=sel.value; save(); draw(); });
      mGrp.appendChild(sel);
      bar.appendChild(mGrp);
      // category selector
      if (S.mode==='agg'){
        const cGrp = elh('div', {class:'cc-grp'});
        cGrp.appendChild(elh('span', {class:'cc-lbl'}, 'categories'));
        cGrp.appendChild(chip('all', false, null, ()=>{ DATA.groups.forEach(c=>S.cats.add(c)); save(); draw(); }));
        cGrp.appendChild(chip('none', false, null, ()=>{ S.cats.clear(); save(); draw(); }));
        if (DATA.groups.some(c=>exclude.has(c)))
          cGrp.appendChild(chip('values only', false, null, ()=>{ S.cats=new Set(DATA.groups.filter(c=>!exclude.has(c))); save(); draw(); }));
        DATA.groups.forEach(c=> cGrp.appendChild(chip(c, S.cats.has(c), null, ()=>{
          S.cats.has(c)?S.cats.delete(c):S.cats.add(c); save(); draw(); })));
        bar.appendChild(cGrp);
      } else if (S.mode==='bucket'){
        const cGrp = elh('div', {class:'cc-grp'});
        cGrp.appendChild(elh('span', {class:'cc-lbl'}, 'buckets'));
        cGrp.appendChild(chip('all', false, null, ()=>{ BUCKETS.forEach(b=>S.buckets.add(b)); save(); draw(); }));
        cGrp.appendChild(chip('none', false, null, ()=>{ S.buckets.clear(); save(); draw(); }));
        BUCKETS.forEach(b=> cGrp.appendChild(chip(b, S.buckets.has(b), null, ()=>{
          S.buckets.has(b)?S.buckets.delete(b):S.buckets.add(b); save(); draw(); })));
        bar.appendChild(cGrp);
      } else {
        const cGrp = elh('div', {class:'cc-grp'});
        cGrp.appendChild(elh('span', {class:'cc-lbl'}, 'category'));
        const dsel = elh('select');
        DATA.groups.forEach(c=>{ const o=elh('option', {value:c}, c); if(c===S.drill)o.setAttribute('selected',''); dsel.appendChild(o); });
        dsel.value=S.drill;
        dsel.addEventListener('change', ()=>{ S.drill=dsel.value; save(); draw(); });
        cGrp.appendChild(dsel);
        bar.appendChild(cGrp);
      }
      // model toggles
      const moGrp = elh('div', {class:'cc-grp'});
      moGrp.appendChild(elh('span', {class:'cc-lbl'}, 'models'));
      moGrp.appendChild(chip('all', false, null, ()=>{ DATA.models.forEach(m=>S.models.add(m)); save(); draw(); }));
      moGrp.appendChild(chip('none', false, null, ()=>{ S.models.clear(); save(); draw(); }));
      DATA.models.forEach(m=> moGrp.appendChild(chip(mlabel(m), S.models.has(m), color(m), ()=>{
        S.models.has(m)?S.models.delete(m):S.models.add(m); save(); draw(); })));
      bar.appendChild(moGrp);
      return bar;
    }

    function caption(){
      let txt;
      if ((DATA.n_runs||1) < 2) txt = 'Single judge pass — no error bars.';
      else if (isJudge(S.metric)) txt = 'Error bars: ±1 SD across ' + DATA.n_runs + ' judge passes.';
      else txt = 'Metric is embedding-based (judge-invariant) — no error bars across the ' + DATA.n_runs + ' passes.';
      const hint = S.mode==='agg'
        ? ' Click a category to break it out by question.'
        : S.mode==='bucket'
        ? ' Click a bucket to break it out into its categories.'
        : ' Click a bar (or its question label) to jump to that question’s responses.';
      const cap = elh('div', {class:'cc-cap'});
      cap.appendChild(elh('b', {}, S.mode==='agg' ? 'Per-category' : S.mode==='bucket' ? 'Per-bucket' : ('Category: '+S.drill)));
      cap.appendChild(document.createTextNode(' · ' + txt + hint));
      return cap;
    }

    function draw(){
      root.innerHTML='';
      root.appendChild(controls());
      const groups = S.mode==='agg' ? aggBy('group',DATA.groups,S.cats)
                   : S.mode==='bucket' ? aggBy('bucket',BUCKETS,S.buckets)
                   : drillGroups();
      if (!groups.length || !groups.some(g=>g.bars.length)){
        root.appendChild(elh('div', {class:'cc-cap'}, 'Nothing selected — pick at least one category and one model.'));
        return;
      }
      root.appendChild(drawChart(groups));
      root.appendChild(caption());
    }
    draw();
  };
})();
"""
