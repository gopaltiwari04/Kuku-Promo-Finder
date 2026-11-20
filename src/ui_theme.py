"""Kuku Promo Finder - simple presentation layer for app.py.

Plain dark theme, one accent colour, a subtle fade as panels scroll in and hover states. No intro animation:
the page shows the plan as soon as the data is loaded.
The Kuku FM logo is never redrawn here: drop the official file into static/ (see find_logo) and it is used
as-is; until then the header shows the plain text "KUKU FM".
Nothing in this module touches the pipeline or raw_outputs/.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------------------------------- palette
BG = "#0B0D12"
CARD = "#12161F"
CARD_2 = "#171C27"
BORDER = "#1F2533"
BORDER_2 = "#2A3142"
TEXT_1 = "#F5F7FA"
TEXT_2 = "#C5CCD8"
TEXT_3 = "#8C96A8"
ACCENT = "#FF5A36"
CRIMSON = "#EF4444"
AMBER = "#F59E0B"
CORAL = ACCENT

WAVE_COLORS = {1: "#10B981", 2: "#3B82F6", 3: "#F59E0B", 4: "#64748B"}
WAVE_NAMES = {1: "Post first", 2: "Post next", 3: "Later", 4: "Hold"}
WAVE_SHORT = {1: "WAVE 1", 2: "WAVE 2", 3: "WAVE 3", 4: "HOLD"}
WAVE_SYMBOLS = {1: "circle", 2: "diamond", 3: "square", 4: "x-thin-open"}
TYPE_COLORS = {"continuous": "#E8552A", "multi_segment_intra": "#8B6CF0", "multi_segment_cross": "#0E9FB6"}
TYPE_NAMES = {"continuous": "One stretch", "multi_segment_intra": "Jump-cut · same episode",
              "multi_segment_cross": "Across episodes"}
TRACK_COLORS = {"CAMERA": "#38BDF8", "ACTION": "#FB923C", "DIALOGUE": "#34D399", "SOUND_FX": "#FBBF24"}
SCORE_LABELS = {"hook_strength": "Hook", "cliffhanger_tension": "Cliffhanger", "standalone_clarity": "Makes sense alone",
                "pacing_density": "Pacing", "trope_virality": "Virality", "spoiler_risk": "Spoiler risk"}

FONT_SANS = "'Plus Jakarta Sans', Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace"


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def wave_of(p_or_wave: Any) -> int:
    w = p_or_wave.get("wave", p_or_wave.get("tier", 4)) if isinstance(p_or_wave, dict) else p_or_wave
    return w if w in WAVE_COLORS else 4


# ===================================================================================== logo
LOGO_DIRS = ("static", "assets", ".")
LOGO_EXTS = (".svg", ".png", ".webp", ".jpg", ".jpeg")
_MIME = {".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def find_logo() -> Optional[Path]:
    """The official Kuku FM logo, if the user has saved it: static/kuku_logo.png (or .svg/.webp/.jpg),
    or any image under static/ or assets/ (subfolders included) whose name contains 'kuku' or 'logo'."""
    for d in LOGO_DIRS:
        folder = ROOT / d
        if not folder.is_dir():
            continue
        files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in LOGO_EXTS)
        for want in ("kuku_logo", "kuku-logo", "kukufm", "kuku", "logo"):
            for f in files:
                if want in f.stem.lower():
                    return f
    return None


def logo_html(height: int = 30) -> str:
    f = find_logo()
    if f is not None:
        data = base64.b64encode(f.read_bytes()).decode("ascii")
        return (f'<img class="kp-logo-img" src="data:{_MIME[f.suffix.lower()]};base64,{data}" alt="Kuku FM" '
                f'style="height:{height}px">')
    return f'<span class="kp-logo-txt" style="font-size:{height * 0.62:.0f}px">KUKU FM</span>'


# ===================================================================================== CSS
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@font-face { font-family: 'Plus Jakarta Sans'; font-style: normal; font-weight: 200 800; font-display: swap;
  src: url('app/static/fonts/plus-jakarta-sans-latin-wght-normal.woff2') format('woff2'); }
@font-face { font-family: 'JetBrains Mono'; font-style: normal; font-weight: 100 800; font-display: swap;
  src: url('app/static/fonts/jetbrains-mono-latin-wght-normal.woff2') format('woff2'); }
html, body, .stApp, .stMarkdown, button, input, textarea, select { font-family: __SANS__; }
.stApp { background: __BG__; color: __T1__; }
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer { display: none !important; }
[data-testid="stMainBlockContainer"] { max-width: 1180px; padding: 88px 2rem 3rem; }
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .kp-fixed) { position: absolute; height: 0; margin: 0; }
p, li { color: __T2__; }
[data-testid="stCaptionContainer"] p { color: __T3__ !important; }
.kp-mono { font-family: __MONO__; }
/* top bar */
.kp-top { position: fixed; top: 0; left: 0; right: 0; height: 60px; z-index: 1000; display: flex; align-items: center; gap: 14px;
  padding: 0 28px; background: rgba(11,13,18,.94); border-bottom: 1px solid __BORDER__; backdrop-filter: blur(10px); }
.kp-logo { display: flex; align-items: center; height: 32px; }
.kp-logo-img { display: block; width: auto; }
.kp-logo-txt { font-weight: 800; letter-spacing: .06em; color: #fff; white-space: nowrap; line-height: 1; }
.kp-top .app { font-weight: 700; color: __T1__; font-size: .98rem; }
.kp-top .sep { width: 1px; height: 20px; background: __BORDER2__; }
.kp-top .show { color: __T3__; font-size: .88rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kp-top .right { margin-left: auto; font-size: .8rem; color: __T3__; white-space: nowrap; }
.kp-top .right b { color: __T2__; font-weight: 600; }
/* intro header */
.kp-hero h1 { font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: __T1__; margin: 6px 0 8px; padding: 0; }
.kp-hero p.lede { font-size: 1rem; line-height: 1.65; color: __T2__; max-width: 820px; margin: 0; }
.kp-hero p.lede b { color: __T1__; }
.kp-rules { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 4px; }
.kp-rule { font-size: .8rem; color: __T2__; padding: 5px 11px; border-radius: 999px; border: 1px solid __BORDER2__; background: __CARD__; }
.kp-rule b { color: __ACCENT__; font-weight: 700; margin-right: 4px; }
.kp-stats { display: flex; flex-wrap: wrap; gap: 0; margin: 20px 0 6px; border: 1px solid __BORDER__; border-radius: 14px; background: __CARD__; overflow: hidden; }
.kp-stat { flex: 1 1 140px; padding: 14px 18px; border-right: 1px solid __BORDER__; }
.kp-stat:last-child { border-right: none; }
.kp-stat .n { font-size: 1.55rem; font-weight: 800; color: __T1__; line-height: 1.1; }
.kp-stat .l { font-size: .78rem; color: __T3__; margin-top: 3px; }
/* section text */
.kp-h2 { font-size: 1.2rem; font-weight: 700; color: __T1__; margin: 8px 0 2px; }
.kp-sub { font-size: .88rem; color: __T3__; margin: 0 0 10px; line-height: 1.55; }
.kp-note { font-size: .8rem; color: __T3__; line-height: 1.5; margin-top: 6px; }
/* panels (keyed bordered containers) */
[class*="st-key-kp_"] { background: __CARD__ !important; border: 1px solid __BORDER__ !important; border-radius: 14px !important;
  padding: 18px 20px !important; transition: border-color .25s, transform .25s; }
[class*="st-key-kp_card"]:hover { border-color: __BORDER2__ !important; transform: translateY(-2px); }
[data-testid="stExpander"] details { border: 1px solid __BORDER__ !important; border-radius: 10px !important; background: rgba(255,255,255,.015) !important; }
/* badges */
.kp-badge { display: inline-block; font-size: .72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; margin-right: 6px; white-space: nowrap; }
.kp-chip { display: inline-block; font-family: __MONO__; font-size: .66rem; font-weight: 600; padding: 2px 7px; border-radius: 6px; white-space: nowrap; }
/* promo card */
.kp-card-h { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.kp-rank { font-size: 1.5rem; font-weight: 800; color: __T1__; min-width: 46px; }
.kp-score { margin-left: auto; text-align: right; }
.kp-score b { font-size: 1.35rem; font-weight: 800; color: __T1__; }
.kp-score span { display: block; font-size: .7rem; color: __T3__; }
.kp-segs { margin: 14px 0 6px; border-top: 1px solid __BORDER__; }
.kp-seg { display: grid; grid-template-columns: 22px 64px 210px 70px 1fr; gap: 12px; align-items: center; padding: 9px 0; border-bottom: 1px solid __BORDER__; }
.kp-seg .i { font-size: .74rem; color: __T3__; }
.kp-seg .ep { font-family: __MONO__; font-size: .8rem; font-weight: 600; color: __T1__; }
.kp-seg .tc { font-family: __MONO__; font-size: .86rem; color: __T1__; }
.kp-seg .d { font-family: __MONO__; font-size: .8rem; color: __T2__; text-align: right; }
.kp-bar { position: relative; height: 6px; border-radius: 4px; background: #232938; }
.kp-bar .span { position: absolute; top: 0; bottom: 0; border-radius: 4px; background: __ACCENT__; }
.kp-bar .hook { position: absolute; top: -2px; bottom: -2px; border-radius: 2px; background: __AMBER__; }
.kp-bar .cut { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #fff; }
.kp-total { font-size: .8rem; color: __T3__; margin: 6px 0 12px; }
.kp-total b { color: __T1__; font-family: __MONO__; }
.kp-line { display: grid; grid-template-columns: 132px 1fr; gap: 14px; padding: 8px 0; }
.kp-line .k { font-size: .76rem; font-weight: 700; letter-spacing: .02em; padding-top: 2px; }
.kp-line .t { font-size: .92rem; color: __T1__; line-height: 1.55; }
.kp-scores { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.kp-sc { font-size: .74rem; color: __T2__; padding: 4px 9px; border-radius: 8px; background: __CARD2__; border: 1px solid __BORDER__; }
.kp-sc b { color: __T1__; font-family: __MONO__; margin-left: 4px; }
.kp-sc.bad b { color: __CRIMSON__; }
.kp-warn { font-size: .8rem; color: #FCD34D; background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.25); border-radius: 8px; padding: 7px 10px; margin-top: 10px; }
/* compact list rows (ranking tab) */
.kp-row { display: grid; grid-template-columns: 44px 230px 1fr; gap: 14px; padding: 11px 0; border-bottom: 1px solid __BORDER__; align-items: start; }
.kp-row .r { font-weight: 800; color: __T1__; font-size: 1rem; }
.kp-row .s { font-family: __MONO__; font-size: .8rem; color: __T1__; line-height: 1.6; }
.kp-row .w { font-size: .88rem; color: __T2__; line-height: 1.5; }
.kp-row .w b { color: __T1__; }
/* failure modes / checks */
.kp-check { display: flex; gap: 10px; align-items: flex-start; font-size: .84rem; color: __T2__; margin: 6px 0; line-height: 1.5; }
.kp-check .ok { flex: 0 0 18px; height: 18px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: .66rem; font-weight: 800; margin-top: 1px; }
.kp-step { display: grid; grid-template-columns: 26px 1fr; gap: 8px; font-size: .88rem; color: __T2__; line-height: 1.55; margin: 7px 0; }
.kp-step .n { font-weight: 800; color: __ACCENT__; }
.kp-tbl { max-height: 420px; overflow: auto; border-radius: 10px; border: 1px solid __BORDER__; }
.kp-tbl table { width: 100%; border-collapse: collapse; font-size: .8rem; }
.kp-tbl th { position: sticky; top: 0; background: __CARD2__; color: __T3__; font-weight: 600; font-size: .72rem; text-align: left; padding: 8px; border-bottom: 1px solid __BORDER__; }
.kp-tbl td { padding: 6px 8px; border-bottom: 1px solid __BORDER__; color: __T1__; vertical-align: top; }
.kp-tbl td.m { font-family: __MONO__; font-size: .72rem; color: __T2__; white-space: nowrap; }
.kp-tbl td.txt { min-width: 320px; line-height: 1.5; }
/* buttons */
div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 600; }
button[data-testid="stBaseButton-primary"] { background: __ACCENT__ !important; border-color: __ACCENT__ !important; }
/* subtle fade as panels scroll in and out */
@supports (animation-timeline: view()) {
  [class*="st-key-kp_"], .kp-fade { animation: kpIn linear both, kpOut linear forwards; animation-timeline: view(), view();
    animation-range: entry 0% entry 55%, exit 45% exit 100%; }
}
@keyframes kpIn { from { opacity: 0; translate: 0 18px; } to { opacity: 1; translate: 0 0; } }
@keyframes kpOut { from { opacity: 1; } to { opacity: .15; } }
@media (max-width: 900px) { .kp-seg { grid-template-columns: 20px 56px 1fr 60px; } .kp-seg .kp-bar { display: none; }
  .kp-row { grid-template-columns: 36px 1fr; } .kp-row .w { grid-column: 1 / -1; } .kp-top .show { display: none; } }
@media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation: none !important; transition: none !important; }
  }
"""

def _fill(css: str) -> str:
    for k, v in {"__SANS__": FONT_SANS, "__MONO__": FONT_MONO, "__BG__": BG, "__CARD__": CARD, "__CARD2__": CARD_2,
                 "__BORDER__": BORDER, "__BORDER2__": BORDER_2, "__T1__": TEXT_1, "__T2__": TEXT_2, "__T3__": TEXT_3,
                 "__ACCENT__": ACCENT, "__AMBER__": AMBER, "__CRIMSON__": CRIMSON}.items():
        css = css.replace(k, v)
    return "\n".join(line for line in css.splitlines() if line.strip())   # no blank lines inside the HTML block


CSS = _fill(_CSS)


def inject_css(st, intro: bool = False) -> None:
    st.markdown(f'<style>{CSS}</style><div class="kp-fixed"></div>',
                unsafe_allow_html=True)


# ===================================================================================== components
def topbar_html(show_title: str, scorer: str) -> str:
    return (f'<div class="kp-fixed"><div class="kp-top"><div class="kp-logo">{logo_html(32)}</div>'
            f'<span class="app">Promo Finder</span><span class="sep"></span><span class="show">{esc(show_title)}</span>'
            f'<span class="right">Scorer: <b>{esc(scorer)}</b></span></div></div>')


def hero_html(show_title: str, n_eps: int, runtime_min: float, n_promos: int, n_wave1: int, checks: str,
              n_plan: int = 0) -> str:
    rules = "".join(f'<span class="kp-rule"><b>{a}</b>{b}</span>' for a, b in (
        ("20–60 s", "clip from the show"), ("0–3 s", "opens on a hook"), ("Last beat", "cuts on a cliffhanger"),
        ("Cold viewer", "makes sense alone")))
    stats = "".join(f'<div class="kp-stat"><div class="n">{n}</div><div class="l">{esc(l)}</div></div>' for n, l in (
        (n_eps, "episodes read"), (n_promos, "promos found"), (n_plan, "in the posting plan"),
        (n_wave1, "to post first"), (checks, "automatic checks passed")))
    return (f'<div class="kp-hero"><h1>Find the promos</h1><p class="lede">Reads every episode screenplay of '
            f'<b>{esc(show_title)}</b> and returns ready-to-cut promos: the episode, the start and end timestamp of '
            f'every segment, the hook, the cliffhanger and why it works, ranked by what to post first.</p>'
            f'<div class="kp-rules">{rules}</div><div class="kp-stats">{stats}</div></div>')


def badge(text: str, color: str) -> str:
    return (f'<span class="kp-badge" style="color:{color};background:{rgba(color, .12)};'
            f'border:1px solid {rgba(color, .35)}">{esc(text)}</span>')


def h2(title: str, sub: str = "") -> str:
    s = f'<div class="kp-sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="kp-fade"><div class="kp-h2">{esc(title)}</div>{s}</div>'


def _ts(ts: str) -> float:
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def segments_html(p: Dict[str, Any], durations: Dict[int, float], hook_sec: float = 3.0) -> str:
    rows = []
    segs = p["segments"]
    for i, s in enumerate(segs):
        a, b = _ts(s["start_timestamp"]), _ts(s["end_timestamp"])
        D = max(float(durations.get(s["episode"]) or 0), b) or b
        bar = (f'<div class="kp-bar" title="where this segment sits in Ep {s["episode"]} ({D:.0f} s)">'
               f'<div class="span" style="left:{a / D * 100:.2f}%;width:{(b - a) / D * 100:.2f}%"></div>')
        if i == 0:
            bar += f'<div class="hook" style="left:{a / D * 100:.2f}%;width:{min(hook_sec, b - a) / D * 100:.2f}%"></div>'
        if i == len(segs) - 1:
            bar += f'<div class="cut" style="left:calc({b / D * 100:.2f}% - 1px)"></div>'
        bar += "</div>"
        rows.append(f'<div class="kp-seg"><span class="i">{i + 1}</span><span class="ep">EP {s["episode"]:02d}</span>'
                    f'<span class="tc">{esc(s["start_timestamp"])} → {esc(s["end_timestamp"])}</span>'
                    f'<span class="d">{s["duration_sec"]:.1f} s</span>{bar}</div>')
    return (f'<div class="kp-segs">{"".join(rows)}</div><div class="kp-total">Total <b>{p["total_duration_sec"]:.1f} s</b>'
            f' · {len(segs)} segment{"s" if len(segs) > 1 else ""} in this order · bar = position in the episode, '
            f'<span style="color:{AMBER}">amber</span> = first 3 s, white = cut</div>')


def promo_card_html(p: Dict[str, Any], durations: Dict[int, float], review: Optional[Dict[str, Any]] = None) -> str:
    w = wave_of(p)
    eps = sorted(set(p["episodes"]))
    ep_txt = f"Ep {eps[0]}" if len(eps) == 1 else "Ep " + " + ".join(str(e) for e in eps)
    status = (review or {}).get("status")
    stat = {"approved": badge("Approved", "#10B981"), "rejected": badge("Rejected", CRIMSON)}.get(status, "")
    sig = p.get("signals") or {}
    src = badge("Sells the premise", "#38BDF8") if sig.get("premise_reveal") else ""
    if sig.get("audience") == "retarget":
        src += badge("Finale: retarget only", TEXT_3)
    head = (f'<div class="kp-card-h"><span class="kp-rank">#{p["post_order"]}</span>'
            f'{badge(WAVE_NAMES[w], WAVE_COLORS[w])}{badge(TYPE_NAMES.get(p["promo_type"], p["promo_type"]), TYPE_COLORS.get(p["promo_type"], TEXT_3))}'
            f'{src}{stat}<span style="color:{TEXT_3};font-size:.82rem">{esc(ep_txt)} · {esc(p["promo_id"])}</span>'
            f'<span class="kp-score"><b>{p["composite_score"]:.1f}</b><span>score / 100</span></span></div>')
    why_here = p.get("post_reason") if w in (1, 2, 3) else p.get("wave_reason")
    order = (f'<div class="kp-note" style="margin-top:6px">{"Posted here because" if w in (1, 2, 3) else "Held because"}: '
             f'{esc(why_here or "")}</div>') if why_here else ""
    lines = "".join(f'<div class="kp-line"><span class="k" style="color:{c}">{k}</span><span class="t">{esc(t)}</span></div>'
                    for k, t, c in (("Hook (0–3 s)", p["hook"], AMBER), ("Cliffhanger", p["cliffhanger"], "#F87171"),
                                    ("Why it works", p["why_it_works"], "#34D399")))
    sc = p.get("scores", {})
    chips = "".join(f'<span class="kp-sc{" bad" if k == "spoiler_risk" else ""}">{SCORE_LABELS[k]}<b>{sc[k]:.1f}</b></span>'
                    for k in SCORE_LABELS if k in sc)
    warn = (f'<div class="kp-warn">Check before posting: {esc("; ".join(p["review_reasons"]))}</div>'
            if p.get("review_reasons") else "")
    return head + order + segments_html(p, durations) + lines + f'<div class="kp-scores">{chips}</div>' + warn


def rank_row_html(p: Dict[str, Any]) -> str:
    segs = "<br>".join(f'EP {s["episode"]:02d} {esc(s["start_timestamp"])}→{esc(s["end_timestamp"])}' for s in p["segments"])
    return (f'<div class="kp-row"><span class="r">#{p["post_order"]}</span><span class="s">{segs}<br>'
            f'<span style="color:{TEXT_3}">{p["total_duration_sec"]:.1f} s · score {p["composite_score"]:.1f}</span></span>'
            f'<span class="w"><b>Why it works:</b> {esc(p["why_it_works"])}<br><span style="color:{TEXT_3}">Posted here '
            f'because: {esc(p.get("post_reason", ""))}</span></span></div>')


def track_table_html(rows: Sequence[Dict[str, Any]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in ("Ep", "Start", "End", "Promo t", "Track", "Who", "Text"))
    body = []
    for r in rows:
        col = TRACK_COLORS.get(r["Track"], TEXT_3)
        chip = f'<span class="kp-chip" style="color:{col};background:{rgba(col, .12)}">{esc(r["Track"].replace("_", " "))}</span>'
        body.append(f'<tr><td class="m">{r["Ep"]}</td><td class="m">{esc(r["Start"])}</td><td class="m">{esc(r["End"])}</td>'
                    f'<td class="m">{esc(r["Promo t"])}</td><td>{chip}</td><td class="m">{esc(r["Speaker / delivery"])}</td>'
                    f'<td class="txt">{esc(r["Text"])}</td></tr>')
    return f'<div class="kp-tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def check_row(ok: bool, name: str, detail: str) -> str:
    col = "#10B981" if ok else CRIMSON
    return (f'<div class="kp-check"><span class="ok" style="color:{col};background:{rgba(col, .14)}">{"✓" if ok else "✗"}</span>'
            f'<span><b style="color:{TEXT_1}">{esc(name.replace("_", " "))}</b> · {esc(detail)}</span></div>')


def steps_html(steps: List[str]) -> str:
    return "".join(f'<div class="kp-step"><span class="n">{i}.</span><span>{esc(s)}</span></div>'
                   for i, s in enumerate(steps, 1))
