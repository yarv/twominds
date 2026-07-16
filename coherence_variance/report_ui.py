"""Shared building blocks for the self-contained HTML reports.

``report.py`` / ``multi_report.py`` / ``families_report.py`` each ship one
portable HTML file; this module holds what they used to copy: the model color
``PALETTE`` (single source, injected into the JS), the JSON-injection guard,
the document shell, the common CSS design system (``BASE_CSS``), and the JS
helper preamble (``BASE_JS``) every report page loads first.

Load order per page: ``BASE_JS`` script → data blob(s) → chart renderer →
page script. ``BASE_JS`` owns the top-level helper names; page scripts and the
chart IIFE must not redeclare them.
"""

from __future__ import annotations

import json

PALETTE = [
    "#4f9dff",
    "#ff8a5c",
    "#5ad19a",
    "#c98bff",
    "#ffd24a",
    "#ff6b9d",
    "#6be0e0",
    "#b0b85a",
    "#e0846b",
    "#8a9bff",
    "#7ad17a",
    "#d99bff",
]

# One banding for the framing-effect ARI everywhere it is displayed
# (report.py's Families tab verdicts and families_report's colored pills):
# < low → framing-invariant, < high → some framing effect, >= high → the
# framing determines the answer.
FAM_ARI_BANDS = (0.10, 0.40)


def fam_verdict(ari: float | None) -> str:
    """Plain-language verdict for a framing-effect ARI, per FAM_ARI_BANDS."""
    if ari is None:
        return "—"
    low, high = FAM_ARI_BANDS
    if abs(ari) >= high:
        return "answer follows the framing"
    if abs(ari) >= low:
        return "some framing effect"
    return "framing-invariant"


def is_family_question(qmeta: dict, qid: str) -> bool:
    """A cross-variant framing-family question. Within-prompt resampling is the
    wrong metric for these (the signal is the cross-variant split, see
    ``families.py`` / ``families_report.html``), so every within-prompt view —
    the explorer cards, the interactive chart, and the static PNG — excludes
    them rather than showing meaningless low-variance bars."""
    return bool((qmeta.get(qid) or {}).get("family"))


def json_blob(x) -> str:
    """JSON for inlining inside a <script> block: '</' must be escaped or a
    '</script>' inside the data would close the tag mid-blob."""
    return json.dumps(x).replace("</", "<\\/")


def script_blob(name: str, x) -> str:
    return f"<script>const {name} = {json_blob(x)};</script>"


def html_document(title: str, css: str, body: str) -> str:
    """The shared document shell; ``body`` includes the page's scripts."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body>
{body}
</body></html>
"""


BASE_CSS = """
:root {
  --bg:#0f1115; --card:#1a1d24; --card2:#13161c; --muted:#8b93a7; --fg:#e6e9ef;
  --line:#2a2f3a; --accent:#4f9dff; --red:#ff6b6b; --amber:#ffd24a;
  --green:#5ad19a; --purple:#c98bff;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); position:sticky; top:0;
  background:rgba(15,17,21,.97); backdrop-filter:blur(6px); z-index:10; }
h1 { font-size:18px; margin:0; }
.dash { color:var(--muted); font-size:12.5px; margin-top:4px; }
.dash b { color:var(--fg); }
.dash .chip { display:inline-block; margin-right:12px; }
.dash .hot { color:var(--red); } .dash .warm { color:var(--amber); }
.controls { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-top:12px; }
.controls label { font-size:11.5px; color:var(--muted); display:flex; gap:5px; align-items:center; }
select, input[type=number], input[type=text] { background:var(--card); color:var(--fg);
  border:1px solid var(--line); border-radius:6px; padding:4px 7px; font-size:12.5px; }
input[type=number] { width:52px; }
input[type=text]#search { width:200px; }
button { background:var(--card); color:var(--fg); border:1px solid var(--line);
  border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer; }
button:hover { border-color:var(--accent); }

#cards { padding:14px 20px; display:flex; flex-direction:column; gap:10px; max-width:1180px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.card.open { border-color:#39404e; }
.card-head { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:11px 14px;
  cursor:pointer; user-select:none; }
.card-head:hover { background:#1e222b; }
.chev { transition:transform .15s; color:var(--muted); font-size:11px; width:10px; }
.card.open .chev { transform:rotate(90deg); }
.tag { font-size:11px; padding:1px 7px; border-radius:10px; background:#2a2f3a; color:var(--muted); }
.tag.model { color:#cfe0ff; background:#22304a; }
.tag.group { color:#d9ffe6; background:#1f3a2c; }
.stats { font-size:11.5px; color:var(--muted); display:flex; gap:12px; flex-wrap:wrap;
  margin-left:auto; align-items:center; }
.stats b { color:var(--fg); }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot.red{background:var(--red);} .dot.amber{background:var(--amber);} .dot.purple{background:var(--purple);}
.dots { display:flex; gap:5px; align-items:center; }

.body { padding:0 14px 12px; border-top:1px solid var(--line); }
.q { color:var(--muted); font-size:12px; white-space:pre-wrap; margin:10px 0; max-height:140px;
  overflow:auto; border-left:2px solid var(--line); padding-left:9px; }
.rationale { font-size:12.5px; margin:8px 0; }
.flags { display:flex; gap:6px; flex-wrap:wrap; margin:6px 0; }
.flag { font-size:11px; padding:1px 8px; border-radius:10px; background:#3a2a2a; color:#ffb3b3; }
.flag.parsefail { background:#3a3520; color:var(--amber); }

.sep { font-size:11px; color:var(--muted); margin:10px 0 4px; display:flex; gap:8px; align-items:center; }
.sep .box { width:11px; height:11px; border-radius:3px; }
.resp { border-left:4px solid; background:var(--card2); border-radius:0 6px 6px 0; margin:5px 0; }
.resp-head { display:flex; gap:9px; align-items:center; padding:6px 10px; cursor:pointer; }
.resp-head:hover { background:#171b22; }
.resp .badge { font-size:10.5px; color:var(--muted); font-family:ui-monospace,monospace; white-space:nowrap; }
.resp .swatch { width:9px; height:9px; border-radius:2px; display:inline-block; }
.resp .snip { color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; flex:1; }
.resp .full { white-space:pre-wrap; font-size:12.5px; padding:2px 12px 10px; }
.empty { color:var(--muted); padding:24px; text-align:center; }

/* banner shown above the cards when the chart focuses one question */
.qfocus { margin:10px 20px 0; max-width:1180px; padding:7px 12px; border-radius:8px;
  background:#1f2a3a; border:1px solid #2f3f57; color:#cfe0ff; font-size:12px;
  display:flex; gap:10px; align-items:center; }
.qfocus b { color:#fff; } .qfocus .x { margin-left:auto; cursor:pointer; color:var(--muted); }
.qfocus .x:hover { color:var(--fg); }
"""

# The JS helper preamble every report page loads before its own script.
# `esc` escapes quotes too: its output is used inside double-quoted
# title="..." attributes, where a bare " would end the attribute.
BASE_JS = (
    f"const PALETTE = {json.dumps(PALETTE)};\n"
    + r"""const $ = (s)=>document.querySelector(s);
const gcolor = (g)=> (g==null||g<0) ? '#666' : PALETTE[g % PALETTE.length];
const color = gcolor;
const esc = (s)=> (s==null?'':String(s)).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (x)=> (x==null||isNaN(x)) ? '–' : Number(x).toFixed(2);
const pct = (x)=> (x==null) ? '–' : Math.round(x*100)+'%';
const trunc = (s,n)=> (s&&s.length>n) ? s.slice(0,n-1)+'…' : (s||'');
// -sum p log p over group frequencies; fallback when metrics.group_entropy is absent (older analyses).
const entropyOf = (labels)=>{ if(!labels||!labels.length) return 0; const c={}; labels.forEach(l=>c[l]=(c[l]||0)+1);
  const n=labels.length; let h=0; for(const k in c){const p=c[k]/n; h-=p*Math.log(p);} return h; };
const NUL = '\x1f';
const mean = (xs)=> xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0;
const pstd = (xs)=>{ if(xs.length<2) return 0; const m=mean(xs); return Math.sqrt(xs.reduce((s,x)=>s+(x-m)*(x-m),0)/xs.length); };
const SVGNS = 'http://www.w3.org/2000/svg';
const svgEl = (t,a,txt)=>{ const e=document.createElementNS(SVGNS,t); for(const k in (a||{})) e.setAttribute(k,a[k]); if(txt!=null) e.textContent=txt; return e; };
const elh = (t,a,txt)=>{ const e=document.createElement(t); for(const k in (a||{})) e.setAttribute(k,a[k]); if(txt!=null) e.textContent=txt; return e; };
// shared grouped-bar pieces: y grid + ticks + rotated axis label, and the ±SD error bar
const ccYAxis = (s, PAD, W, plotH, maxY, label)=>{ const NT=5;
  for (let i=0;i<=NT;i++){
    const v=maxY*i/NT, yy=PAD.t + plotH*(1 - v/maxY);
    s.appendChild(svgEl('line', {x1:PAD.l, y1:yy, x2:W-PAD.r, y2:yy, class:i?'grid':'axis'}));
    s.appendChild(svgEl('text', {x:PAD.l-7, y:yy+4, 'text-anchor':'end', 'font-size':10}, v.toFixed(2)));
  }
  s.appendChild(svgEl('text', {x:14, y:PAD.t+plotH/2, 'text-anchor':'middle', 'font-size':11,
    transform:'rotate(-90 14 '+(PAD.t+plotH/2)+')'}, label)); };
const ccErrorBar = (s, cx, yhi, ylo, bw)=>{ const cap=Math.min(5,bw/3), ec='#e7ebf3';
  s.appendChild(svgEl('line', {x1:cx, y1:yhi, x2:cx, y2:ylo, stroke:ec, 'stroke-width':1.3}));
  s.appendChild(svgEl('line', {x1:cx-cap, y1:yhi, x2:cx+cap, y2:yhi, stroke:ec, 'stroke-width':1.3}));
  s.appendChild(svgEl('line', {x1:cx-cap, y1:ylo, x2:cx+cap, y2:ylo, stroke:ec, 'stroke-width':1.3})); };
// Filter/sort state persisted per report (localStorage) + shareable (URL hash).
// Object-typed defaults stay out of the hash: they don't survive a string roundtrip.
const stateStore = (SKEY, DEFAULTS)=>({
  load(){
    let s = {};
    try { s = JSON.parse(localStorage.getItem(SKEY) || '{}'); } catch(e) {}
    const h = new URLSearchParams(location.hash.slice(1));
    for (const k of Object.keys(DEFAULTS)){
      const d = DEFAULTS[k];
      if (!h.has(k) || typeof d==='object') continue;
      const v = h.get(k);
      s[k] = (typeof d==='boolean') ? v==='1' : (typeof d==='number') ? Number(v) : v;
    }
    return Object.assign({}, DEFAULTS, s);
  },
  save(state){
    try { localStorage.setItem(SKEY, JSON.stringify(state)); } catch(e) {}
    const h = new URLSearchParams();
    for (const k of Object.keys(DEFAULTS)){
      const v = state[k], d = DEFAULTS[k];
      if (v === d || typeof d==='object') continue;
      h.set(k, typeof v==='boolean' ? (v?'1':'0') : String(v));
    }
    const hs = h.toString();
    try { history.replaceState(null, '', hs ? ('#'+hs) : (location.pathname+location.search)); } catch(e) {}
  },
});
"""
)
