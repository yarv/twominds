"""Self-contained HTML report for eyeballing response variance.

One shareable file (data blob + inlined CSS/JS, matching the repo's report
convention), organised into four tabs so the headline lands before the detail:

The report speaks plain language: "answer set" (one model's N answers to one
question, a.k.a. a bundle in the code), "distinct positions" (judge groups),
"answer spread" (group entropy), "cross-check agreement" (judge-vs-embedding
ARI), "wording variety" (mean pairwise cosine distance). Jargon lives only in
hover tooltips and the Method tab's glossary, sourced from the single JS
``GLOSSARY`` map.

* **Overview** — a two-sentence method intro, stat tiles with direction hints
  (higher/lower is better), auto-generated takeaway bullets, the interactive
  per-category chart, and a per-model comparison table (row click jumps to the
  answers filtered to that model).
* **Models** — per-model drill-down: category breakdown + the model's least
  consistent questions, each linking to its answers.
* **Answers** (id ``explorer``) — the full card browser. Cards (one per model x
  question) are collapsible; judged self-contradictions auto-expand (flagged
  cards stay collapsed — flags can be plentiful judge side-notes). Controls: model / category / bucket / embedding-backend filters,
  answer + card sorts, min-positions + min-clusters thresholds, free-text
  search, flag filter, toggles, expand/collapse all, and a dashboard
  summarising the filtered set.
* **Method & setup** (id ``setup``) — what ran, how answers were collected,
  how consistency was scored, a glossary, cost, the question roster, and
  provenance.

Filter/sort state (including the active tab) persists in the URL hash + a per-run
localStorage key (so independent reports never share state), with a "reset
filters" button and an auto-reset-to-defaults guard so a transferred report never
opens blank. All data needed is already in ``analysis.json`` (per-response
``judge_labels`` / cluster labels, ``agreement`` ARI, ``metrics``, and — for runs
generated after the manifest gained it — ``model_display`` / ``config``).
"""

from __future__ import annotations

import json
from pathlib import Path

from twominds import category_chart
from twominds.report_ui import (
    BASE_CSS,
    BASE_JS,
    fam_verdict,
    html_document,
    json_blob,
)

# Report-specific styles on top of report_ui.BASE_CSS.
_CSS = """
.legend { display:flex; gap:10px; flex-wrap:wrap; font-size:11px; color:var(--muted); margin:6px 0 2px; }
.legend .sw { display:inline-flex; gap:4px; align-items:center; }
.legend .box { width:11px; height:11px; border-radius:3px; display:inline-block; }
.resp-actions { font-size:11px; color:var(--muted); margin:6px 0; }
.resp-actions a { color:var(--accent); cursor:pointer; margin-right:10px; }

/* tabs */
nav.tabs { display:flex; gap:6px; margin-top:10px; }
nav.tabs button { border-radius:8px 8px 0 0; border-bottom:none; padding:6px 16px;
  font-size:13px; color:var(--muted); background:transparent; }
nav.tabs button.on { background:var(--card); color:var(--fg); border-color:#39404e;
  border-bottom:2px solid var(--accent); }
section.tab { display:none; }
section.tab.active { display:block; }
.pane { padding:14px 20px; max-width:1180px; }
.famtab { border-collapse:collapse; margin-top:10px; font-size:13px; }
.famtab th, .famtab td { text-align:left; padding:6px 14px 6px 0; border-bottom:1px solid var(--line); }
.famtab th { color:var(--muted); font-weight:500; font-size:11.5px; }
.fambtn { display:inline-block; margin-top:12px; padding:8px 14px; border:1px solid var(--accent);
  border-radius:8px; color:var(--accent); text-decoration:none; }
.fambtn:hover { background:rgba(79,157,255,.12); }
.pane h2 { font-size:14px; color:var(--fg); margin:18px 0 8px; }
.pane h2:first-child { margin-top:4px; }

/* overview stat tiles */
.tiles { display:flex; gap:10px; flex-wrap:wrap; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:10px 16px; min-width:110px; max-width:220px; }
.tile .v { font-size:20px; font-weight:600; }
.tile .v.hot { color:var(--red); } .tile .v.warm { color:var(--amber); }
.tile .k { font-size:12px; margin-top:2px; }
.tile .sub { font-size:10.5px; color:var(--muted); margin-top:2px; line-height:1.35; }

/* plain-language helpers */
.note { color:var(--muted); font-size:12.5px; max-width:900px; margin:0 0 12px; line-height:1.55; }
.note b { color:var(--fg); }
.hint { color:var(--muted); font-size:11.5px; font-weight:400; }
.tbl th[title] { cursor:help; text-decoration:underline dotted #3a4150; text-underline-offset:3px; }
.chip[title], .stats span[title] { cursor:help; }

.takeaways { margin:12px 0 0; padding:10px 14px; background:var(--card2);
  border:1px solid var(--line); border-radius:10px; font-size:12.5px; }
.takeaways li { margin:3px 0; }
.takeaways .num { color:var(--accent); }

/* summary / setup tables */
table.tbl { border-collapse:collapse; font-size:12.5px; width:100%; }
.tbl th { text-align:left; color:var(--muted); font-weight:500; font-size:11.5px;
  padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
.tbl td { padding:6px 10px; border-bottom:1px solid #20242e; }
.tbl tr.click { cursor:pointer; }
.tbl tr.click:hover td { background:#1e222b; }
.tbl td.num, .tbl th.num { text-align:right; font-variant-numeric:tabular-nums; }
.tbl .muted { color:var(--muted); }
.tbl .mono { font-family:ui-monospace,monospace; font-size:11.5px; }
.tbl a { color:var(--accent); cursor:pointer; }
.tblwrap { overflow-x:auto; border:1px solid var(--line); border-radius:10px;
  background:var(--card); }
.kv { font-size:12.5px; }
.kv .k { color:var(--muted); display:inline-block; min-width:140px; }
.kv div { padding:2px 0; }
"""

_JS = r"""
const cardKey = (r)=> r.model + NUL + r.question_id;
const respKey = (r,i)=> cardKey(r) + NUL + i;

// Per-run key so independent reports (different runs / judge-runs) never share
// filter state — important when several of these self-contained HTMLs are opened
// from the same browser, since all local files share one localStorage origin.
const REPORT_ID = (DATA.run_dir || 'variance') + (DATA.judge_run ? '::' + DATA.judge_run : '');
const SKEY = 'variance_report_state_v1:' + REPORT_ID;
// Embeddings are optional (-b none = judge-only run): with no backends, every
// embedding-derived affordance (clusters, cross-check, variety, backend picker)
// is hidden rather than shown blank.
const EMB = (DATA.backends || []).length > 0;
const DEFAULTS = { tab:'overview', model:'__all__', group:'__all__', bucket:'__all__', backend:DATA.primary_backend,
  rsort:'original', csort:'incoherent', minG:1, minC:1, search:'', flag:'__any__',
  onlyContra:false, onlyFlag:false, question:'__all__' };
// derived from the header so optional tabs (e.g. Families) just work
const TABS = Array.from(document.querySelectorAll('nav.tabs button'), b=>b.dataset.tab);
// Readable model label (roster display / original provider id); short name is
// the key everywhere, display is the descriptive form shown in tables/tooltips.
const displayName = (m)=> (DATA.model_display||{})[m] || m;
let STATE = Object.assign({}, DEFAULTS);
const openCards = new Set();   // card keys currently expanded
const openResps = new Set();   // response keys currently expanded

const STORE = stateStore(SKEY, DEFAULTS);
const loadState = ()=> STORE.load();
const saveState = ()=> STORE.save(STATE);

const divergence = (r)=>{ const a = r.agreement && r.agreement[STATE.backend];
  return (a && a.ari!=null) ? (1 - a.ari) : 0; };
const nClusters = (r)=>{ const c = r.clusters && r.clusters[STATE.backend]; return c ? c.n_clusters : 0; };

function passes(r){
  const j = r.judge || {};
  // Cross-variant framing-family bundles aren't within-prompt-coherence signal
  // (their signal is the cross-variant split — see the families report link).
  if ((DATA.questions[r.question_id]||{}).family) return false;
  if (STATE.model!=='__all__' && r.model!==STATE.model) return false;
  if (STATE.group!=='__all__' && r.group!==STATE.group) return false;
  if (STATE.bucket!=='__all__' && (DATA.questions[r.question_id]||{}).bucket!==STATE.bucket) return false;
  if (STATE.question!=='__all__' && r.question_id!==STATE.question) return false;
  if (STATE.onlyContra && !j.contradiction) return false;
  if (STATE.onlyFlag && !(j.flags && j.flags.length)) return false;
  if ((j.n_groups||0) < STATE.minG) return false;
  if (EMB && nClusters(r) < STATE.minC) return false;
  if (STATE.flag!=='__any__' && !((j.flags||[]).includes(STATE.flag))) return false;
  if (STATE.search){
    const q = STATE.search.toLowerCase();
    const hay = [].concat(r.responses||[], [j.rationale||''], j.flags||[],
      [(DATA.questions[r.question_id]||{}).prompt||'']).join('\n').toLowerCase();
    if (!hay.includes(q)) return false;
  }
  return true;
}

function sortCards(rows){
  const ng=(r)=>(r.judge||{}).n_groups||0, cd=(r)=>(r.metrics||{}).mean_pairwise_cosine_dist||0;
  const tie=(a,b)=> a.model.localeCompare(b.model)||a.question_id.localeCompare(b.question_id);
  const cmps = {
    incoherent:(a,b)=> ng(b)-ng(a) || cd(b)-cd(a) || tie(a,b),
    cosdist:(a,b)=> cd(b)-cd(a) || tie(a,b),
    divergence:(a,b)=> divergence(b)-divergence(a) || tie(a,b),
    model:(a,b)=> a.model.localeCompare(b.model)||a.group.localeCompare(b.group)||a.question_id.localeCompare(b.question_id),
    group:(a,b)=> a.group.localeCompare(b.group)||a.model.localeCompare(b.model)||a.question_id.localeCompare(b.question_id),
  };
  return rows.slice().sort(cmps[STATE.csort] || tie);
}

function respOrder(r){
  const n = r.responses.length, idx = Array.from({length:n}, (_,i)=>i);
  if (STATE.rsort==='original') return idx.map(i=>[i, null]);
  const labels = STATE.rsort==='judge'
    ? (r.judge_labels || idx)
    : (((r.clusters||{})[STATE.backend]||{}).labels || idx);
  idx.sort((a,b)=> (labels[a]-labels[b]) || (a-b));
  return idx.map(i=>[i, labels[i]]);
}

function renderCard(r){
  const key = cardKey(r), isOpen = openCards.has(key), j = r.judge||{}, m = r.metrics||{};
  const agr = (r.agreement && r.agreement[STATE.backend]) || {};
  const ng = j.n_groups, nc = nClusters(r), div = divergence(r);
  let dots = '';
  if (j.contradiction) dots += '<span class="dot red" title="self-contradiction: two answers take incompatible positions"></span>';
  if (j.flags && j.flags.length) dots += '<span class="dot amber" title="flagged: the judge noted something unusual"></span>';
  if (div >= 0.5 && (ng>1 || nc>1)) dots += '<span class="dot purple" title="the judge and the embedding cross-check disagree here — read with care"></span>';

  let h = '<div class="card'+(isOpen?' open':'')+'">';
  h += '<div class="card-head" data-card="'+esc(key)+'">'
     + '<span class="chev">▶</span>'
     + '<span class="tag model">'+esc(r.model)+'</span>'
     + '<span class="tag group">'+esc(r.group)+'</span>'
     + '<span class="tag">'+esc(r.question_id)+'</span>'
     + '<span class="dots">'+dots+'</span>'
     + '<span class="stats">'
       + '<span>answers=<b>'+m.n+'</b></span>'
       + '<span title="'+esc(GLOSSARY.positions)+'">positions=<b>'+(ng??'–')+'</b></span>'
       + (EMB ? '<span title="independent embedding clustering ('+esc(STATE.backend)+' backend)">clusters=<b>'+nc+'</b></span>'
              + '<span title="'+esc(GLOSSARY.crosscheck)+'">cross-check=<b>'+fmt(agr.ari)+'</b></span>'
              + '<span title="'+esc(GLOSSARY.variety)+'">variety=<b>'+fmt(m.mean_pairwise_cosine_dist)+'</b></span>' : '')
       + '<span title="'+esc(GLOSSARY.spread)+'">spread=<b>'+fmt(m.group_entropy!=null ? m.group_entropy : entropyOf(r.judge_labels))+'</b></span>'
     + '</span></div>';

  if (isOpen){
    h += '<div class="body">';
    h += '<div class="q">'+esc((DATA.questions[r.question_id]||{}).prompt || r.question_id)+'</div>';
    if (j.rationale) h += '<div class="rationale">'+esc(j.rationale)+'</div>';
    if (j.flags && j.flags.length)
      h += '<div class="flags">'+j.flags.map(f=>'<span class="flag '+(f==='judge_parse_failed'?'parsefail':'')+'">'+esc(f)+'</span>').join('')+'</div>';
    // legend (judge positions present)
    if (ng){
      let leg = '';
      for (let g=0; g<ng; g++) leg += '<span class="sw"><span class="box" style="background:'+color(g)+'"></span>position '+(g+1)+'</span>';
      h += '<div class="legend" title="'+esc(GLOSSARY.positions)+'">'+leg+'</div>';
    }
    h += '<div class="resp-actions"><a data-expand="'+esc(key)+'">expand all answers</a>'
       + '<a data-collapse="'+esc(key)+'">collapse all</a></div>';

    // counts per label for separators
    const order = respOrder(r);
    const counts = {};
    if (STATE.rsort!=='original') order.forEach(([i,l])=>{ counts[l]=(counts[l]||0)+1; });
    let lastLabel = undefined;
    for (const [i, label] of order){
      if (STATE.rsort!=='original' && label!==lastLabel){
        lastLabel = label;
        const kind = STATE.rsort==='judge' ? 'Position' : 'Cluster';
        h += '<div class="sep"><span class="box" style="background:'+color(label)+'"></span>'
           + kind+' '+(label+1)+' — '+counts[label]+' answer'+(counts[label]===1?'':'s')+'</div>';
      }
      h += renderResp(r, i);
    }
    h += '</div>';
  }
  h += '</div>';
  return h;
}

function renderResp(r, i){
  const text = r.responses[i] || '', rk = respKey(r,i), open = openResps.has(rk);
  const jg = (r.judge_labels||[])[i] ?? -1;
  const cl = (((r.clusters||{})[STATE.backend]||{}).labels||[])[i] ?? -1;
  const snip = text.replace(/\s+/g,' ').slice(0,140);
  let h = '<div class="resp" style="border-left-color:'+color(jg)+'">';
  h += '<div class="resp-head" data-resp="'+esc(rk)+'">'
     + '<span class="badge">#'+(i+1)+'</span>'
     + '<span class="swatch" style="background:'+color(jg)+'" title="position (judge)"></span>'
     + (EMB
        ? '<span class="badge" title="position (judge) · cluster (embeddings) · length">'
          + (jg<0?'–':'p'+(jg+1))+' · '+(cl<0?'–':'c'+(cl+1))+' · '+text.length+'ch</span>'
        : '<span class="badge" title="position (judge) · length">'
          + (jg<0?'–':'p'+(jg+1))+' · '+text.length+'ch</span>')
     + (open ? '' : '<span class="snip">'+esc(snip)+'</span>')
     + '</div>';
  if (open) h += '<div class="full">'+esc(text)+'</div>';
  h += '</div>';
  return h;
}

function render(){
  const rows = sortCards(DATA.results.filter(passes));
  // focus banner: set by clicking a question bar in the category chart
  if (STATE.question!=='__all__'){
    const qp = (DATA.questions[STATE.question]||{}).prompt || STATE.question;
    $('#qfocus').style.display='flex';
    $('#qfocus').innerHTML = 'Showing the answers to <b>'+esc(STATE.question)+'</b> — '
      + esc(qp.length>120?qp.slice(0,119)+'…':qp)
      + '<span class="x" id="qfocusClear" title="show all questions">✕ clear</span>';
  } else { $('#qfocus').style.display='none'; $('#qfocus').innerHTML=''; }
  $('#cards').innerHTML = rows.length
    ? rows.map(renderCard).join('')
    : '<div class="empty">No answers match the current filters.</div>';
  // dashboard
  const nc = rows.filter(r=>(r.judge||{}).contradiction).length;
  const nf = rows.filter(r=>((r.judge||{}).flags||[]).length).length;
  const mg = rows.length ? (rows.reduce((s,r)=>s+((r.judge||{}).n_groups||0),0)/rows.length) : 0;
  // Mean grouping entropy across the (filtered) question bank — weights by
  // group sizes, so an 11-vs-1 split scores near 0 and a 6-vs-6 split maximal,
  // which mean judge groups (a bare count) can't tell apart. Prefer the stored
  // per-bundle metrics.group_entropy; fall back to recomputing from labels.
  const mh = rows.length ? (rows.reduce((s,r)=>{
      const m=r.metrics||{}; const h=(m.group_entropy!=null)?m.group_entropy:entropyOf(r.judge_labels);
      return s+(h||0); },0)/rows.length) : 0;
  const aris = rows.map(r=>(r.agreement&&r.agreement[STATE.backend]||{}).ari).filter(x=>x!=null);
  const ma = aris.length ? aris.reduce((a,b)=>a+b,0)/aris.length : null;
  $('#dash').innerHTML =
      '<span class="chip"><b>'+rows.length+'</b> answer sets shown</span>'
    + '<span class="chip hot" title="'+esc(GLOSSARY.contra)+'"><b>'+nc+'</b> self-contradictions</span>'
    + '<span class="chip warm" title="'+esc(GLOSSARY.flagged)+'"><b>'+nf+'</b> flagged</span>'
    + '<span class="chip" title="'+esc(GLOSSARY.positions)+'">avg. positions <b>'+fmt(mg)+'</b></span>'
    + '<span class="chip" title="'+esc(GLOSSARY.spread)+'">avg. spread <b>'+fmt(mh)+'</b></span>'
    + (EMB ? '<span class="chip" title="'+esc(GLOSSARY.crosscheck)+'">cross-check agreement <b>'+fmt(ma)+'</b></span>' : '');
}

// ---------- tabs ----------
function renderTabs(){
  if (!TABS.includes(STATE.tab)) STATE.tab = 'overview';
  document.querySelectorAll('nav.tabs button').forEach(b=>
    b.classList.toggle('on', b.dataset.tab===STATE.tab));
  TABS.forEach(t=>{ const s=document.getElementById('tab-'+t);
    if (s) s.classList.toggle('active', t===STATE.tab); });
}
function gotoTab(t){ STATE.tab=t; saveState(); renderTabs(); }

// Jump into the explorer with a model (and optionally a question) pre-filtered.
function exploreModel(m, qid){
  STATE = Object.assign({}, DEFAULTS, {tab:'explorer', backend:STATE.backend, model:m});
  if (qid){ STATE.question = qid;
    DATA.results.forEach(r=>{ if (r.question_id===qid && r.model===m) openCards.add(cardKey(r)); }); }
  syncControls(); saveState(); renderTabs(); render();
  window.scrollTo({top:0});
}

// ---------- aggregation (overview + models tabs) ----------
// Within-prompt bundles only: framing-family bundles measure a different thing
// (the cross-variant split) and are excluded from the explorer for the same reason.
const COHERENT_ROWS = DATA.results.filter(r=>!((DATA.questions[r.question_id]||{}).family));
const HAS_JUDGE = COHERENT_ROWS.some(r=>r.judge);
const bundleH = (r)=>{ const m=r.metrics||{};
  return (m.group_entropy!=null) ? m.group_entropy : (r.judge_labels ? entropyOf(r.judge_labels) : null); };

function aggRows(rows){
  const nc = rows.filter(r=>(r.judge||{}).contradiction).length;
  const nf = rows.filter(r=>((r.judge||{}).flags||[]).length).length;
  const gs = rows.map(r=>(r.judge||{}).n_groups).filter(x=>x!=null);
  const hs = rows.map(bundleH).filter(x=>x!=null);
  const aris = rows.map(r=>(r.agreement&&r.agreement[DATA.primary_backend]||{}).ari).filter(x=>x!=null);
  const cds = rows.map(r=>(r.metrics||{}).mean_pairwise_cosine_dist).filter(x=>x!=null);
  const avg = (xs)=> xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : null;
  // share of judged answer sets where every answer took the same position
  const consist = gs.length ? gs.filter(g=>g===1).length/gs.length : null;
  return { n:rows.length, contra:nc, flagged:nf, consist,
    meanG:avg(gs), meanH:avg(hs), meanARI:avg(aris), meanCD:avg(cds) };
}
const byModel = ()=> DATA.models.map(m=>({model:m, ...aggRows(COHERENT_ROWS.filter(r=>r.model===m))}));

// ---------- shared column definitions (plain labels + explanatory tooltips) --
const GLOSSARY = {
  positions: 'How many genuinely different positions the answers took, according to the judge. 1 = the model said the same thing every time.',
  consist: 'Share of answer sets (one model × one question) where every answer took the same position. Higher is better.',
  contra: 'Answer sets where two answers took logically incompatible positions. Lower is better.',
  flagged: 'Answer sets where the judge noted something unusual (refusals, identity confusion, striking content).',
  spread: 'How evenly answers split across positions (entropy). 0 = every answer agrees; ~0.69 = two equal halves; higher = more or evener positions. Lower is better.',
  crosscheck: 'Agreement between the judge’s grouping and an independent embedding clustering (adjusted Rand index: 1 = identical, 0 = unrelated). Where this is low, treat the numbers with extra care.',
  variety: 'Average embedding distance between answers — higher means answers are worded/structured more differently. Style variation, not necessarily disagreement.'
};
// Column set for aggregate tables (overview per-model, models-tab per-category).
function aggCols(extended){
  const cols = [
    {label:'questions', tip:'answer sets covered (one per question)', cell:(a)=>a.n},
  ];
  if (HAS_JUDGE) cols.push(
    {label:'fully consistent', tip:GLOSSARY.consist, cell:(a)=>pct(a.consist)},
    {label:'self-contradictions', tip:GLOSSARY.contra, cell:(a)=>a.contra},
    {label:'flagged', tip:GLOSSARY.flagged, cell:(a)=>a.flagged},
    {label:'avg. positions', tip:GLOSSARY.positions, cell:(a)=>fmt(a.meanG)},
  );
  if (extended){
    if (HAS_JUDGE) cols.push(
      {label:'answer spread', tip:GLOSSARY.spread, cell:(a)=>fmt(a.meanH)},
    );
    if (HAS_JUDGE && EMB) cols.push(
      {label:'cross-check', tip:GLOSSARY.crosscheck, cell:(a)=>fmt(a.meanARI)},
    );
    if (EMB) cols.push({label:'wording variety', tip:GLOSSARY.variety, cell:(a)=>fmt(a.meanCD)});
  }
  return cols;
}
function aggTable(firstCol, rows, extended){
  const cols = aggCols(extended);
  let h = '<table class="tbl"><tr><th>'+firstCol+'</th>'
    + cols.map(c=>'<th class="num" title="'+esc(c.tip)+'">'+c.label+'</th>').join('') + '</tr>';
  for (const r of rows){
    h += r.rowOpen + cols.map(c=>'<td class="num">'+c.cell(r.agg)+'</td>').join('') + '</tr>';
  }
  return h + '</table>';
}

// ---------- overview ----------
function renderOverview(){
  const a = aggRows(COHERENT_ROWS);
  const qids = new Set(COHERENT_ROWS.map(r=>r.question_id));
  const nAnswers = (DATA.config||{}).n
    || (COHERENT_ROWS.length ? (COHERENT_ROWS[0].metrics||{}).n : null);

  // What this report is — orient readers who don't know the method.
  $('#intro').innerHTML =
    'Each model answered every question <b>'+(nAnswers??'N')+'&times;</b>, with the identical prompt each time '
    + '(sampling temperature 1.0), so any differences between the answers come from the model itself. '
    + (HAS_JUDGE
        ? 'A separate judge model read each set of answers and grouped them by the position they take — '
          + 'a self-consistent model takes the same position every time.'
        : 'The judge step was skipped on this run, so only embedding-based measures (how differently the answers are worded) are available.');

  const tile = (v,k,sub,cls)=>'<div class="tile"><div class="v '+(cls||'')+'">'+v+'</div>'
    +'<div class="k">'+k+'</div>'+(sub?'<div class="sub">'+sub+'</div>':'')+'</div>';
  let t = tile(DATA.models.length,'models','compared side by side')
    + tile(qids.size,'questions','each asked '+(nAnswers??'N')+'× per model');
  if (HAS_JUDGE){
    t += tile(pct(a.consist),'fully consistent','answer sets where every answer agrees · higher is better',
              (a.consist!=null && a.consist<0.5)?'warm':'')
      + tile(a.contra,'self-contradictions','sets with logically incompatible answers · lower is better', a.contra?'hot':'')
      + tile(a.flagged,'flagged','sets with unusual answers (refusals etc.)', a.flagged?'warm':'');
  } else {
    t += tile(fmt(a.meanCD),'wording variety','average embedding distance between answers');
  }
  $('#tiles').innerHTML = t;

  // auto-takeaways, in plain language
  const rows = byModel().filter(r=>r.n);
  const bullets = [];
  if (HAS_JUDGE && rows.some(r=>r.consist!=null)){
    const worst = rows.slice().sort((x,y)=>(x.consist??2)-(y.consist??2))[0];
    const best = rows.slice().sort((x,y)=>(y.consist??-1)-(x.consist??-1))[0];
    bullets.push('Least consistent model: <b>'+esc(worst.model)+'</b> — every answer agreed on only '
      +'<span class="num">'+pct(worst.consist)+'</span> of its '+worst.n+' questions'
      +(worst.contra?' ('+worst.contra+' self-contradiction'+(worst.contra===1?'':'s')+')':'')+'.');
    if (best.model!==worst.model)
      bullets.push('Most consistent: <b>'+esc(best.model)+'</b> ('+pct(best.consist)+' of questions answered consistently).');
  }
  const groups = [...new Set(COHERENT_ROWS.map(r=>r.group))].filter(Boolean);
  if (HAS_JUDGE && groups.length>1){
    const ga = groups.map(g=>({g, ...aggRows(COHERENT_ROWS.filter(r=>r.group===g))}))
      .filter(x=>x.consist!=null).sort((x,y)=>x.consist-y.consist);
    if (ga.length) bullets.push('Models waver most on <b>'+esc(ga[0].g)+'</b> questions '
      +'(consistent on '+pct(ga[0].consist)+' there'
      +(ga[0].contra?', '+ga[0].contra+' self-contradictions':'')+').');
  }
  if (HAS_JUDGE){
    const div = rows.filter(r=>r.meanARI!=null).sort((x,y)=>x.meanARI-y.meanARI)[0];
    if (div && div.meanARI!=null && div.meanARI<0.7)
      bullets.push('Caveat: for <b>'+esc(div.model)+'</b> the automated cross-check (embedding clustering) '
        +'often disagrees with the judge’s grouping (agreement '+fmt(div.meanARI)+' of 1) — read its numbers with extra care.');
  }
  $('#takeaways').innerHTML = bullets.length
    ? '<ul style="margin:4px 0 4px 18px;padding:0">'+bullets.map(b=>'<li>'+b+'</li>').join('')+'</ul>' : '';
  $('#takeaways').style.display = bullets.length ? 'block' : 'none';

  // per-model summary table (hover the column headers for what each measures)
  const tableRows = rows.map(r=>({
    agg: r,
    rowOpen: '<tr class="click" data-m="'+esc(r.model)+'" title="'+esc(displayName(r.model))+' — click to explore">'
      + '<td><b>'+esc(r.model)+'</b>'
      + (displayName(r.model)!==r.model ? ' <span class="muted">'+esc(displayName(r.model))+'</span>' : '')
      + '</td>',
  }));
  $('#modelTable').innerHTML = '<div class="tblwrap">'+aggTable('model', tableRows, false)+'</div>';
  $('#modelTable').querySelectorAll('tr.click').forEach(tr=>
    tr.addEventListener('click', ()=> exploreModel(tr.dataset.m)));
}

// ---------- models tab ----------
function renderModelDetail(){
  const m = $('#mmodel').value;
  const rows = COHERENT_ROWS.filter(r=>r.model===m);
  const groups = [...new Set(rows.map(r=>r.group))].filter(Boolean).sort();
  let h = '<div class="kv" style="margin-bottom:10px"><div><span class="k">model</span><b>'+esc(m)+'</b></div>'
    + (displayName(m)!==m ? '<div><span class="k">full id / source</span>'+esc(displayName(m))+'</div>' : '')
    + '</div>';
  h += '<h2>By question category <span class="hint">— hover a column header for what it measures</span></h2>';
  const tableRows = groups.map(g=>({
    agg: aggRows(rows.filter(r=>r.group===g)),
    rowOpen: '<tr><td>'+esc(g)+'</td>',
  }));
  h += '<div class="tblwrap">'+aggTable('category', tableRows, true)+'</div>';

  const worst = rows.slice().sort((x,y)=>{
    const gx=((y.judge||{}).n_groups||0)-((x.judge||{}).n_groups||0);
    return gx || ((bundleH(y)||0)-(bundleH(x)||0));
  }).slice(0,5);
  if (worst.length){
    h += '<h2>Least consistent questions</h2><div class="tblwrap"><table class="tbl">'
      + '<tr><th>question</th><th>category</th>'
      + '<th class="num" title="'+esc(GLOSSARY.positions)+'">distinct positions</th>'
      + '<th class="num" title="'+esc(GLOSSARY.spread)+'">answer spread</th>'
      + '<th class="num" title="'+esc(GLOSSARY.variety)+'">wording variety</th><th></th></tr>';
    for (const r of worst){
      const q=(DATA.questions[r.question_id]||{}).prompt||r.question_id;
      h += '<tr><td title="'+esc(q)+'">'+esc(r.question_id)
        +' <span class="muted">'+esc(q.length>70?q.slice(0,69)+'…':q)+'</span></td>'
        +'<td>'+esc(r.group)+'</td>'
        +'<td class="num">'+(((r.judge||{}).n_groups)??'–')+'</td>'
        +'<td class="num">'+fmt(bundleH(r))+'</td>'
        +'<td class="num">'+fmt((r.metrics||{}).mean_pairwise_cosine_dist)+'</td>'
        +'<td><a data-q="'+esc(r.question_id)+'">read the answers →</a></td></tr>';
    }
    h += '</table></div>';
  }
  $('#modelDetail').innerHTML = h;
  $('#modelDetail').querySelectorAll('a[data-q]').forEach(a=>
    a.addEventListener('click', ()=> exploreModel(m, a.dataset.q)));
}

// ---------- method & setup tab ----------
function renderSetup(){
  const cfg = DATA.config || {};
  const kv = (k,v)=> (v==null||v==='') ? '' : '<div><span class="k">'+k+'</span>'+v+'</div>';

  // What ran — the models under test.
  let h = '<h2>What ran</h2><div class="tblwrap"><table class="tbl">'
    + '<tr><th>name</th><th>full id / source</th><th>API model string</th><th>reasoning effort</th></tr>';
  const cfgModels = cfg.models || {};
  for (const m of DATA.models){
    const mc = cfgModels[m] || {};
    h += '<tr><td><b>'+esc(m)+'</b></td><td>'+esc(mc.display||displayName(m))+'</td>'
      + '<td class="mono">'+esc(mc.inspect_model||'?')+'</td>'
      + '<td>'+esc(mc.reasoning_effort||'–')+'</td></tr>';
  }
  h += '</table></div>';

  // How the answers were collected.
  const nq = Object.keys(DATA.questions||{}).length;
  h += '<h2>How the answers were collected</h2>'
    + '<p class="note">Every model got the identical prompt for each question, '
    + (cfg.n!=null ? '<b>'+cfg.n+'</b> times' : 'multiple times')
    + (cfg.temperature!=null ? ' at sampling temperature '+cfg.temperature : '')
    + '. Nothing varies between the repeats, so differences between the answers reflect the model, not the setup.</p>'
    + '<div class="kv">'
    + kv('questions', nq || null)
    + kv('answers per question', cfg.n)
    + kv('sampling temperature', cfg.temperature)
    + kv('max tokens per answer', cfg.max_tokens)
    + '</div>';

  // How consistency was scored.
  h += '<h2>How consistency was scored</h2>';
  if (DATA.judge){
    h += '<p class="note">A separate judge model read all answers to a question at once and partitioned '
      + 'them into groups that take the same position ("distinct positions"). It also flagged '
      + 'self-contradictions and anything unusual.'
      + (EMB ? ' As an independent cross-check, the answers were '
             + 'embedded and clustered without the judge; where the two disagree, the numbers deserve more scrutiny.'
             : ' Embedding clustering was disabled on this run (-b none), so all measures come from the judge.')
      + '</p>';
  } else {
    h += '<p class="note">The judge step was skipped on this run — only the embedding-based measures are available.</p>';
  }
  h += '<div class="kv">'
    + kv('judge model', esc(DATA.judge||'(skipped)'))
    + kv('judge reasoning effort', esc(DATA.judge_reasoning||''))
    + kv('embedding cross-check', esc(EMB ? (DATA.backends||[]).join(', ') : '(disabled)'))
    + (EMB ? kv('clustering threshold', DATA.threshold) : '')
    + '</div>';

  // Glossary — the one place every term is defined.
  h += '<h2>Glossary</h2><div class="kv">'
    + kv('answer set', 'the '+(cfg.n!=null?cfg.n+' ':'')+'answers one model gave to one question — the unit most numbers are counted over')
    + kv('distinct positions', esc(GLOSSARY.positions))
    + kv('fully consistent', esc(GLOSSARY.consist))
    + kv('self-contradiction', esc(GLOSSARY.contra))
    + kv('flagged', esc(GLOSSARY.flagged))
    + kv('answer spread', esc(GLOSSARY.spread))
    + (EMB ? kv('cross-check agreement', esc(GLOSSARY.crosscheck))
           + kv('wording variety', esc(GLOSSARY.variety)) : '')
    + '</div>';

  // Cost.
  const cost = DATA.cost || {};
  if (cost.judge || cost.generation){
    h += '<h2>Cost</h2><div class="kv">';
    if (cost.generation){
      const gd = Object.values(cost.generation).reduce((s,v)=>s+((v||{}).dollars||0),0);
      h += kv('collecting answers (est.)', '$'+gd.toFixed(2));
    }
    if (cost.judge){ const j=cost.judge;
      h += kv('judging (est.)', j.est_dollars!=null ? '$'+Number(j.est_dollars).toFixed(2) : null)
        + kv('judging (billed, OpenRouter)', j.openrouter_delta!=null ? '$'+Number(j.openrouter_delta).toFixed(2) : null)
        + kv('judge tokens', (j.in_tok||0).toLocaleString()+' in / '+(j.out_tok||0).toLocaleString()+' out');
    }
    h += '</div>';
  }

  // The questions themselves.
  const qids = Object.keys(DATA.questions||{}).sort();
  if (qids.length){
    h += '<h2>Questions ('+qids.length+')</h2><div class="tblwrap"><table class="tbl">'
      + '<tr><th>id</th><th>category</th><th>bucket</th><th>family</th><th>prompt</th></tr>';
    for (const qid of qids){
      const q = DATA.questions[qid]||{};
      const p = q.prompt||'';
      h += '<tr><td class="mono">'+esc(qid)+'</td><td>'+esc(q.group||'')+'</td>'
        + '<td>'+esc(q.bucket||'')+'</td><td>'+esc(q.family||'')+'</td>'
        + '<td title="'+esc(p)+'">'+esc(p.length>110?p.slice(0,109)+'…':p)+'</td></tr>';
    }
    h += '</table></div>';
  }

  // Provenance — where the data lives on disk.
  h += '<h2>Provenance</h2><div class="kv">'
    + kv('run directory', '<span class="mono">'+esc(DATA.run_dir||'')+'</span>')
    + ((DATA.source_runs&&DATA.source_runs.length) ? kv('source runs', esc(DATA.source_runs.join(', '))) : '')
    + '</div>';
  if (DATA.merge_warnings && DATA.merge_warnings.length)
    h += '<div class="takeaways" style="margin-top:8px">merge warnings: '+esc(DATA.merge_warnings.join(' · '))+'</div>';

  $('#setupBody').innerHTML = h;
}

function opts(sel, vals, withAll){
  const el = $(sel); el.innerHTML = '';
  if (withAll) el.append(new Option('(all)', '__all__'));
  vals.forEach(v=> el.append(new Option(v, v)));
}

function syncControls(){
  for (const [id,k] of [['#model','model'],['#group','group'],['#bucket','bucket'],['#backend','backend'],
      ['#respSort','rsort'],['#cardSort','csort'],['#flagFilter','flag']])
    $(id).value = STATE[k];
  $('#minGroups').value = STATE.minG; $('#minClusters').value = STATE.minC;
  $('#search').value = STATE.search;
  $('#onlyContra').checked = STATE.onlyContra; $('#onlyFlag').checked = STATE.onlyFlag;
}

function wire(){
  const bind = (id, k, ev, get)=> $(id).addEventListener(ev, ()=>{ STATE[k]=get(); saveState(); render(); });
  bind('#model','model','change', ()=>$('#model').value);
  bind('#group','group','change', ()=>$('#group').value);
  bind('#bucket','bucket','change', ()=>$('#bucket').value);
  bind('#backend','backend','change', ()=>$('#backend').value);
  bind('#respSort','rsort','change', ()=>$('#respSort').value);
  bind('#cardSort','csort','change', ()=>$('#cardSort').value);
  bind('#flagFilter','flag','change', ()=>$('#flagFilter').value);
  bind('#minGroups','minG','input', ()=>Math.max(1, Number($('#minGroups').value)||1));
  bind('#minClusters','minC','input', ()=>Math.max(1, Number($('#minClusters').value)||1));
  bind('#search','search','input', ()=>$('#search').value.trim());
  bind('#onlyContra','onlyContra','change', ()=>$('#onlyContra').checked);
  bind('#onlyFlag','onlyFlag','change', ()=>$('#onlyFlag').checked);

  const visibleKeys = ()=> sortCards(DATA.results.filter(passes)).map(cardKey);
  $('#expandAll').addEventListener('click', ()=>{ visibleKeys().forEach(k=>openCards.add(k)); render(); });
  $('#collapseAll').addEventListener('click', ()=>{ openCards.clear(); render(); });
  $('#reset').addEventListener('click', ()=>{ STATE = Object.assign({}, DEFAULTS); syncControls(); saveState(); render(); });

  $('#cards').addEventListener('click', (e)=>{
    const rh = e.target.closest('.resp-head');
    if (rh){ const k=rh.dataset.resp; openResps.has(k)?openResps.delete(k):openResps.add(k); render(); return; }
    const ea = e.target.closest('[data-expand]');
    if (ea){ const ck=ea.dataset.expand; DATA.results.forEach(r=>{ if(cardKey(r)===ck) r.responses.forEach((_,i)=>openResps.add(respKey(r,i))); }); render(); return; }
    const ca = e.target.closest('[data-collapse]');
    if (ca){ const ck=ca.dataset.collapse; DATA.results.forEach(r=>{ if(cardKey(r)===ck) r.responses.forEach((_,i)=>openResps.delete(respKey(r,i))); }); render(); return; }
    const ch = e.target.closest('.card-head');
    if (ch){ const k=ch.dataset.card; openCards.has(k)?openCards.delete(k):openCards.add(k); render(); }
  });
  $('#qfocus').addEventListener('click', (e)=>{
    if (e.target.id==='qfocusClear'){ STATE.question='__all__'; saveState(); render(); }
  });
  document.querySelectorAll('nav.tabs button').forEach(b=>
    b.addEventListener('click', ()=> gotoTab(b.dataset.tab)));
  $('#mmodel').addEventListener('change', renderModelDetail);
}

// Jump the reader from a chart bar to that question's actual response cards:
// filter the cards to the clicked question, expand the relevant bundle(s), and
// scroll the first match into view. info.model set => land on that exact bundle.
function focusQuestion(info){
  if (!info || !info.qid) return;
  STATE.question = info.qid;
  STATE.tab = 'explorer';
  DATA.results.forEach(r=>{ if (r.question_id===info.qid && (!info.model || r.model===info.model)) openCards.add(cardKey(r)); });
  saveState(); renderTabs(); render();
  const want = info.model ? (info.model + NUL + info.qid) : null;
  let target = null;
  document.querySelectorAll('#cards .card-head').forEach(h=>{
    if (target) return;
    if (want ? h.dataset.card===want : (h.dataset.card||'').endsWith(NUL+info.qid)) target = h;
  });
  (target || $('#cards')).scrollIntoView({behavior:'smooth', block:'start'});
}

function init(){
  STATE = loadState();
  // family-tagged bundles never render here (their signal is cross-variant —
  // see the Families tab), so don't offer groups/buckets that would be empty
  const shown = DATA.results.filter(r=>!(DATA.questions[r.question_id]||{}).family);
  const groups = [...new Set(shown.map(r=>r.group))].filter(Boolean);
  const buckets = [...new Set(shown.map(r=>(DATA.questions[r.question_id]||{}).bucket))].filter(Boolean).sort();
  const flags = [...new Set(DATA.results.flatMap(r=>(r.judge||{}).flags||[]))].sort();
  opts('#model', DATA.models.slice(), true);
  opts('#mmodel', DATA.models.slice(), false);
  opts('#group', groups, true);
  opts('#bucket', buckets, true);
  opts('#backend', DATA.backends, false);
  $('#flagFilter').innerHTML = '';
  $('#flagFilter').append(new Option('(any flag)', '__any__'));
  flags.forEach(f=> $('#flagFilter').append(new Option(f, f)));
  if (!DATA.backends.includes(STATE.backend)) STATE.backend = DATA.primary_backend;
  if (!DATA.models.includes(STATE.model)) STATE.model = '__all__';
  if (!EMB){
    // Judge-only run: hide the embedding-dependent controls and sort options.
    for (const sel of ['#backend','#minClusters']){
      const el = $(sel); const lab = el && el.closest('label');
      if (lab) lab.style.display = 'none';
    }
    const kill = (sel, vals)=>{ const el=$(sel);
      [...el.options].forEach(o=>{ if (vals.includes(o.value)) o.remove(); }); };
    kill('#respSort', ['cluster']);
    kill('#cardSort', ['divergence','cosdist']);
    if (STATE.rsort==='cluster') STATE.rsort='original';
    if (['divergence','cosdist'].includes(STATE.csort)) STATE.csort='incoherent';
  }
  syncControls();
  // Safety net: if restored/shared filter state would hide every bundle (stale state
  // from another report, or a shared #hash that doesn't match this dataset), reset to
  // defaults so a transferred report never opens to a blank "no bundles match" screen.
  if (DATA.results.filter(passes).length === 0){
    STATE = Object.assign({}, DEFAULTS);
    syncControls(); saveState();
  }
  wire();
  if (window.initCategoryChart) initCategoryChart(CHART, 'cchart', {storageKey: SKEY, onBarClick: focusQuestion});
  // auto-expand only judged self-contradictions on first load — flags can be
  // plentiful (judge side-notes), and 100+ pre-opened cards helps nobody
  DATA.results.forEach(r=>{ const j=r.judge||{}; if (j.contradiction) openCards.add(cardKey(r)); });
  renderTabs();
  renderOverview();
  renderModelDetail();
  renderSetup();
  render();
}
init();
"""


def _write_sibling_png(analysis: dict, out_path: Path) -> None:
    """Best-effort paper-ready sibling PNG (the static matplotlib figure). The
    interactive in-page chart is the primary view now; this just keeps a
    paper-figure artefact next to the report. Never raises — the static figure is
    a nicety and must not break the (load-bearing) HTML report."""
    try:
        from twominds import category_bars as cb

        metric = cb.default_metric(analysis)
        if metric is None:
            return
        cb.save_png(
            analysis,
            out_path.with_name(f"category_{metric}_bars.png"),
            metric,
            run_label=out_path.resolve().parent.name,
        )
    except Exception:
        pass


def _families_tab_html(analysis: dict) -> tuple[str, str]:
    """(nav button, tab section) surfacing the cross-variant framing families.

    The framing-family signal lives in ``families_report.html`` (built into the
    same dir whenever the analysis has families); the main within-prompt report
    excludes those questions from its chart/cards, so give the signal its own
    tab — a per-(family, model) summary plus a prominent link to the full
    interactive report. ("", "") when there are no families.
    """
    import html as html_mod

    fams = analysis.get("families") or []
    if not fams:
        return "", ""
    n_fams = len({r.get("family") for r in fams})
    top = max(
        (abs((r.get("judge") or {}).get("ari") or 0.0) for r in fams), default=0.0
    )
    rows = []
    for rec in sorted(
        fams, key=lambda r: (r.get("family") or "", r.get("model") or "")
    ):
        judge = rec.get("judge") or {}
        ari = judge.get("ari")
        swing = (rec.get("scalar") or {}).get("swing")
        cells = [
            html_mod.escape(str(rec.get("title") or rec.get("family") or "?")),
            html_mod.escape(str(rec.get("model") or "?")),
            f"{ari:.2f}" if ari is not None else "—",
            f"{swing:.2f}" if swing is not None else "—",
            fam_verdict(ari),
        ]
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    button = '<button data-tab="families">Families</button>'
    section = f"""
<section class="tab" id="tab-families">
  <div class="pane">
    <h2>Framing families <span class="hint">— the same question asked under
    answer-irrelevant framings: does the answer follow the framing?</span></h2>
    <p class="note">{n_fams} framing famil{"y" if n_fams == 1 else "ies"}.
    Framing effect = agreement (ARI) between the judge's answer-groups and the
    framing labels: 0&nbsp;≈ framing-invariant, 1&nbsp;≈ the framing determines
    the answer (strongest here: {top:.2f} of 1). Swing = spread of the
    per-framing scalar answer. These framing families are kept out of the
    within-prompt numbers in the other tabs.</p>
    <table class="famtab">
      <thead><tr><th>family</th><th>model</th><th>framing effect (ARI)</th>
      <th>swing</th><th>read</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <p><a class="fambtn" href="families_report.html">Open the full interactive
    families report — per-variant answers, contingency tables →</a></p>
  </div>
</section>"""
    return button, section


def build_report(analysis: dict, out_path: Path) -> Path:
    out_path = Path(out_path)
    data_json = json_blob(analysis)
    n_models = len(analysis.get("models", []))
    n_questions = len(analysis.get("questions") or {}) or len(
        {r.get("question_id") for r in analysis.get("results", [])}
    )
    n_answers = (analysis.get("config") or {}).get("n")
    per_q = f" × {n_answers} answers each" if n_answers else ""
    judge = analysis.get("judge") or "(none)"
    backends = ", ".join(analysis.get("backends", [])) or "none (judge-only)"
    _write_sibling_png(analysis, out_path)
    chart_data = json_blob(category_chart.build_chart_data(analysis))
    chart_section = category_chart.chart_section_html("cchart")
    fam_button, fam_section = _families_tab_html(analysis)
    body = f"""<header>
  <h1>How consistently do these models answer?</h1>
  <div class="dash" style="margin-top:2px">{n_models} models · {n_questions} questions{per_q} ·
    judge: {judge} · embeddings: {backends}</div>
  <nav class="tabs">
    <button data-tab="overview">Overview</button>
    <button data-tab="models">Models</button>
    <button data-tab="explorer">Answers</button>{fam_button}
    <button data-tab="setup">Method &amp; setup</button>
  </nav>
</header>

<section class="tab" id="tab-overview">
  <div class="pane">
    <p class="note" id="intro"></p>
    <div class="tiles" id="tiles"></div>
    <div class="takeaways" id="takeaways" style="display:none"></div>
  </div>
  {chart_section}
  <div class="pane">
    <h2>How the models compare <span class="hint">— hover a column header for what it measures; click a row to read that model's answers</span></h2>
    <div id="modelTable"></div>
  </div>
</section>

<section class="tab" id="tab-models">
  <div class="pane">
    <div class="controls" style="margin-top:0">
      <label>model <select id="mmodel"></select></label>
    </div>
    <div id="modelDetail" style="margin-top:12px"></div>
  </div>
</section>

<section class="tab" id="tab-explorer">
  <div class="pane" style="padding-bottom:0">
    <div class="dash" id="dash"></div>
    <div class="controls">
      <label>model <select id="model"></select></label>
      <label>category <select id="group"></select></label>
      <label>bucket <select id="bucket"></select></label>
      <label title="which embedding backend's clusters and cross-check to show">embedding backend <select id="backend"></select></label>
      <label>sort answers
        <select id="respSort">
          <option value="original">original order</option>
          <option value="judge">by position (judge)</option>
          <option value="cluster">by cluster (embeddings)</option>
        </select></label>
      <label>sort cards
        <select id="cardSort">
          <option value="incoherent">least consistent first</option>
          <option value="divergence">judge vs cross-check disagreement</option>
          <option value="cosdist">wording variety</option>
          <option value="model">model</option>
          <option value="group">category</option>
        </select></label>
      <label title="show only answer sets with at least this many distinct positions">min positions <input type="number" id="minGroups" min="1" value="1"></label>
      <label title="show only answer sets with at least this many embedding clusters">min clusters <input type="number" id="minClusters" min="1" value="1"></label>
      <label>search <input type="text" id="search" placeholder="text in answers/flags"></label>
      <label>flag <select id="flagFilter"></select></label>
      <label><input type="checkbox" id="onlyContra"> self-contradictions only</label>
      <label><input type="checkbox" id="onlyFlag"> flagged only</label>
      <button id="expandAll">expand all</button>
      <button id="collapseAll">collapse all</button>
      <button id="reset" title="restore all filters and sorts to their defaults">reset filters</button>
    </div>
  </div>
  <div id="qfocus" class="qfocus" style="display:none"></div>
  <div id="cards"></div>
</section>

{fam_section}

<section class="tab" id="tab-setup">
  <div class="pane" id="setupBody"></div>
</section>

<script>{BASE_JS}</script>
<script>const CHART = {chart_data};</script>
<script>{category_chart.CHART_JS}</script>
<script>const DATA = {data_json};</script>
<script>{_JS}</script>"""
    out_path.write_text(
        html_document(
            "Response variance — coherence",
            BASE_CSS + _CSS + category_chart.CHART_CSS,
            body,
        )
    )
    return out_path


def build_report_from_run(run_dir: Path, out_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    analysis = json.loads((run_dir / "analysis.json").read_text())
    out_path = Path(out_path) if out_path else (run_dir / "report.html")
    return build_report(analysis, out_path)
