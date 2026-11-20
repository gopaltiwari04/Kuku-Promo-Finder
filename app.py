"""Kuku Promo Finder - review UI for Problem 1 ("Find the Promos").

    streamlit run app.py

Four tabs, one per part of the brief:
  1. Promos                - the posting plan first: episode + start/end of every segment, hook, cliffhanger, why it
                             works. An editor approves/rejects, nudges the in-point or the cut by one clean block,
                             and downloads the approved cut list.
  2. What to post first    - the score formula, the posting rules in plain words, and the three lists with reasons
  3. Where it picks badly  - measured quality: the independent check, scorer agreement, guardrail effects, rule
                             checks, human labels, failure modes, flagged promos, automatic checks
  4. Try a new screenplay  - upload / paste a screenplay, swap the scoring model, run the same pipeline

Deployment: works without the (confidential) episode folder. The public repo has no raw_outputs/; set the secret
raw_outputs_url (a private link to a zip of raw_outputs/) so the app shows the saved plan, and app_password to
require a password.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src import config
from src import ui_theme as ui
from src.parser import EpisodeIndex, load_corpus, ts_to_sec
from src.pipeline import run_folder, run_on_episodes, run_texts


def _page_icon():
    logo = ui.find_logo()
    if logo is not None and logo.suffix.lower() != ".svg":
        try:
            from PIL import Image
            return Image.open(logo)
        except Exception:
            pass
    return "🎬"


st.set_page_config(page_title="Kuku · Promo Finder", layout="wide", initial_sidebar_state="collapsed",
                   page_icon=_page_icon())


def _ver(v: str):
    return tuple(int("".join(ch for ch in part if ch.isdigit()) or 0) for part in v.split(".")[:2])


WIDE = {"width": "stretch"} if _ver(st.__version__) >= (1, 46) else {"use_container_width": True}
VALIGN = {"vertical_alignment": "bottom"} if _ver(st.__version__) >= (1, 36) else {}
OUT = config.DEFAULT_OUTPUT_DIR
HAVE_EPISODES = config.DEFAULT_INPUT_DIR is not None and Path(config.DEFAULT_INPUT_DIR).is_dir()
PROVIDERS = {"heuristic (free, no API key)": "heuristic", "gemini-3.5-flash-lite": "gemini", "gemini-2.5-flash": "gemini",
             "gemini-2.5-pro": "gemini", "gpt-4o-mini": "openai", "claude (Sonnet)": "anthropic"}
PROVIDER_MODEL = {"gemini-3.5-flash-lite": "gemini-3.5-flash-lite", "gemini-2.5-pro": "gemini-2.5-pro",
                  "gemini-2.5-flash": "gemini-2.5-flash"}
VIEW_OPTIONS = {"Posting plan (all three lists)": (1, 2, 3), "Post first": (1,), "Post next": (2,), "Later": (3,),
                "Held": (4,), "All promos": (1, 2, 3, 4)}
TYPE_OPTIONS = {"All types": None, "One stretch": "continuous", "Jump-cut, same episode": "multi_segment_intra",
                "Across episodes": "multi_segment_cross"}
REVIEW_FILE = config.PROJECT_ROOT / "review" / "decisions.json"

# ------------------------------------------------------------------------------------- optional password
def _secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name.upper()) or None


_PW = _secret("app_password")
if _PW and not st.session_state.get("_authed"):
    ui.inject_css(st)
    st.markdown(ui.topbar_html("", "-"), unsafe_allow_html=True)
    st.markdown(ui.h2("Kuku Promo Finder", "This prototype shows screenplay text from Kuku's input pack, so it is "
                      "password-protected."), unsafe_allow_html=True)
    if st.text_input("Password", type="password", key="pw") == _PW:
        st.session_state["_authed"] = True
        st.rerun()
    st.stop()

ui.inject_css(st)


# ------------------------------------------------------------------------------------- data
@st.cache_resource(show_spinner=False)
def corpus(input_dir: str):
    eps, errs = load_corpus(Path(input_dir))
    return {e.episode_number: EpisodeIndex(e) for e in eps}, errs


@st.cache_data(show_spinner=False)
def load_outputs(stamp: float) -> Dict[str, Any]:
    data = json.loads((OUT / "promos_all_episodes.json").read_text(encoding="utf-8"))
    read = lambda n: json.loads((OUT / n).read_text(encoding="utf-8")) if (OUT / n).exists() else {}
    return {"promos": data["promos"], "meta": data["metadata"], "audit": read("quality_audit_and_bad_picks.json"),
            "stats": pd.read_csv(OUT / "episode_summary_stats.csv"), "verify": read("verification_report.json"),
            "quality": read("quality_eval.json")}


def fetch_outputs() -> bool:
    """The public repo carries no raw_outputs/ (they quote Kuku's screenplay). On a deployed app, the secret
    raw_outputs_url (a private link to a zip of raw_outputs/; a Google Drive share link works) restores them."""
    url = _secret("raw_outputs_url")
    if not url:
        return False
    import re
    import urllib.request
    import zipfile
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=download&)?id=)([\w-]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            OUT.mkdir(parents=True, exist_ok=True)
            for n in z.namelist():
                name = Path(n).name                       # flat: no paths from the zip are trusted
                if name and Path(name).suffix in (".json", ".csv", ".md"):
                    (OUT / name).write_bytes(z.read(n))
    except Exception as exc:
        st.warning(f"Could not load the saved outputs from raw_outputs_url: {exc}")
        return False
    return (OUT / "promos_all_episodes.json").exists()


def ensure_outputs() -> None:
    if (OUT / "promos_all_episodes.json").exists() or fetch_outputs():
        return
    if not HAVE_EPISODES:
        st.error("No raw_outputs/ and no episode folder: run `python run_pipeline.py` locally first.")
        st.stop()
    with st.spinner("No raw_outputs/ yet - running the pipeline on all episodes (~10 s) ..."):
        run_folder(config.DEFAULT_INPUT_DIR, OUT)


def track_rows(index: Dict[int, EpisodeIndex], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every screenplay block (camera, action, dialogue, sound) that falls inside the promo's segments."""
    rows: List[Dict[str, Any]] = []
    offset = 0.0
    for s in segments:
        ix = index.get(s["episode"])
        if ix is None:
            continue
        a, b = ts_to_sec(s["start_timestamp"]), ts_to_sec(s["end_timestamp"])
        for ev in ix.events_in(a, b):
            if ev.track == "DIALOGUE" and (ev.start_sec < a - 1e-6 or ev.end_sec > b + 1e-6):
                continue
            who = ""
            if ev.track == "DIALOGUE":
                who = ev.speaker_raw + (f" ({', '.join(ev.emotion_tags)})" if ev.emotion_tags else "")
            rows.append({"Ep": s["episode"], "Start": ev.start_ts, "End": ev.end_ts,
                         "Promo t": f"+{max(0.0, ev.start_sec - a) + offset:05.2f}s", "Track": ev.track,
                         "Speaker / delivery": who, "Text": ev.text})
        offset += b - a
    return rows


# ------------------------------------------------------------------------------------- editor review
def _sig(p: Dict[str, Any]) -> str:
    return "|".join(f"{s['episode']}:{s['start_timestamp']}-{s['end_timestamp']}" for s in p["segments"])


def load_reviews() -> Dict[str, Any]:
    try:
        return json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_reviews() -> None:
    try:
        REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_FILE.write_text(json.dumps(st.session_state["reviews"], indent=2), encoding="utf-8")
    except Exception:
        pass            # read-only deployments keep decisions for the session only


def review_of(p: Dict[str, Any]) -> Dict[str, Any]:
    rv = st.session_state["reviews"].get(p["promo_id"])
    return rv if rv and rv.get("sig") == _sig(p) else {}


def set_review(p: Dict[str, Any], **kw) -> None:
    rv = dict(review_of(p)) or {"sig": _sig(p), "status": "pending", "start_shift": 0, "end_shift": 0, "note": ""}
    rv.update(kw)
    st.session_state["reviews"][p["promo_id"]] = rv
    save_reviews()


def adjusted(p: Dict[str, Any], index: Dict[int, EpisodeIndex], rv: Dict[str, Any]
             ) -> Tuple[Dict[str, Any], Optional[str]]:
    """Apply the editor's nudges (whole clean blocks) to the in-point and the final cut; refuse invalid cuts."""
    ds, de = int(rv.get("start_shift", 0)), int(rv.get("end_shift", 0))
    if not (ds or de):
        return p, None
    segs = copy.deepcopy(p["segments"])
    first, last = segs[0], segs[-1]
    try:
        starts = [ts for _, ts in index[first["episode"]].clean_starts]
        ends = [ts for _, ts in index[last["episode"]].clean_ends]
        i = starts.index(first["start_timestamp"]) + ds
        j = ends.index(last["end_timestamp"]) + de
    except (KeyError, ValueError):
        return p, "needs the episode files to move cut points"
    if not (0 <= i < len(starts) and 0 <= j < len(ends)):
        return p, "no clean cut point further that way"
    first["start_timestamp"], last["end_timestamp"] = starts[i], ends[j]
    for s in segs:
        s["duration_sec"] = round(ts_to_sec(s["end_timestamp"]) - ts_to_sec(s["start_timestamp"]), 2)
    total = round(sum(s["duration_sec"] for s in segs), 2)
    if any(s["duration_sec"] <= 0 for s in segs) or not (config.MIN_DURATION <= total <= config.MAX_DURATION):
        return p, f"that cut would be {total:.1f} s (must stay {config.MIN_DURATION:.0f}-{config.MAX_DURATION:.0f} s)"
    q = dict(p, segments=segs, total_duration_sec=total)
    return q, None


def cut_list_csv(promos: List[Dict[str, Any]], index: Dict[int, EpisodeIndex]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["post_order", "promo_id", "wave", "segment", "episode", "in", "out", "segment_sec", "total_sec",
                "hook", "cliffhanger", "why_it_works", "edited_by_editor", "editor_note"])
    for p in sorted(promos, key=lambda x: x["post_order"]):
        rv = review_of(p)
        if rv.get("status") != "approved":
            continue
        q, _ = adjusted(p, index, rv)
        for n, s in enumerate(q["segments"], 1):
            w.writerow([p["post_order"], p["promo_id"], p.get("wave"), n, s["episode"], s["start_timestamp"],
                        s["end_timestamp"], f"{s['duration_sec']:.2f}", f"{q['total_duration_sec']:.2f}", p["hook"],
                        p["cliffhanger"], p["why_it_works"], "yes" if q is not p else "no", rv.get("note", "")])
    return buf.getvalue().encode("utf-8")


def promo_card(p: Dict[str, Any], index: Dict[int, EpisodeIndex], durations: Dict[int, float], key: str,
               reviewable: bool = True) -> None:
    rv = review_of(p) if reviewable else {}
    shown, err = adjusted(p, index, rv) if rv else (p, None)
    with st.container(border=True, key=f"kp_card_{key}"):
        st.markdown(ui.promo_card_html(shown, durations, rv), unsafe_allow_html=True)
        if shown is not p:
            st.caption(f"Cut adjusted by the editor: in-point {rv.get('start_shift', 0):+d} block(s), "
                       f"final cut {rv.get('end_shift', 0):+d} block(s). Re-check the first 3 s on the video.")
        if err:
            st.warning(err)
        if reviewable:
            c = st.columns([1, 1, 0.75, 0.75, 0.75, 0.75, 2.4], **VALIGN)
            if c[0].button("Approve", key=f"{key}_ok", type="primary" if rv.get("status") == "approved" else "secondary", **WIDE):
                set_review(p, status="approved")
                st.rerun()
            if c[1].button("Reject", key=f"{key}_no", **WIDE):
                set_review(p, status="rejected")
                st.rerun()
            for col, lab, fld, d in ((c[2], "◀ in", "start_shift", -1), (c[3], "in ▶", "start_shift", 1),
                                     (c[4], "◀ cut", "end_shift", -1), (c[5], "cut ▶", "end_shift", 1)):
                if col.button(lab, key=f"{key}_{fld}_{d}", disabled=not index, help="move by one clean screenplay block", **WIDE):
                    trial = dict(rv or {"start_shift": 0, "end_shift": 0})
                    trial[fld] = int(trial.get(fld, 0)) + d
                    _, e = adjusted(p, index, trial)
                    if e:
                        st.toast(e)
                    else:
                        set_review(p, **{fld: trial[fld]})
                        st.rerun()
            note = c[6].text_input("Editor note", value=rv.get("note", ""), key=f"{key}_note", label_visibility="collapsed",
                                   placeholder="note for the edit (optional)")
            if note != rv.get("note", "") and (rv or note):
                set_review(p, note=note)
        if index:
            with st.expander("Show the screenplay lines in this promo"):
                rows = track_rows(index, shown["segments"])
                st.markdown(ui.track_table_html(rows) if rows else "Source episode not loaded.", unsafe_allow_html=True)


def promos_table(promos: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Post order": p["post_order"], "Wave": ui.WAVE_NAMES[ui.wave_of(p)],
        "Type": ui.TYPE_NAMES.get(p["promo_type"], p["promo_type"]),
        "Segments (episode in → out)": "  +  ".join(f"Ep {s['episode']} {s['start_timestamp']}→{s['end_timestamp']}"
                                                  for s in p["segments"]),
        "Length (s)": round(p["total_duration_sec"], 1), "Hook": p["hook"], "Cliffhanger": p["cliffhanger"],
        "Why it works": p["why_it_works"], "Score": p["composite_score"],
        "Why here / why held": p.get("post_reason") if ui.wave_of(p) < 4 else p.get("wave_reason"),
        "Review": review_of(p).get("status", "pending"), "ID": p["promo_id"]} for p in promos])


def show_promos(promos: List[Dict[str, Any]], index, durations, key: str, max_ep: int, page_size: int = 10,
                reviewable: bool = True) -> None:
    c = st.columns([1.7, 1.4, 1.5, 1.6, 1], **VALIGN)
    waves = VIEW_OPTIONS[c[0].selectbox("Show", list(VIEW_OPTIONS), key=f"{key}_wave")]
    ptype = TYPE_OPTIONS[c[1].selectbox("Promo type", list(TYPE_OPTIONS), key=f"{key}_type")]
    hi_ep = max(max_ep, 2)
    lo, hi = c[2].slider("Episodes", 1, hi_ep, (1, hi_ep), key=f"{key}_eps")
    q = c[3].text_input("Search hook / cliffhanger", key=f"{key}_q", placeholder="e.g. slap, wedding, secret")
    view = c[4].radio("View", ["Cards", "Table"], horizontal=True, key=f"{key}_view")
    shown = sorted([p for p in promos if ui.wave_of(p) in waves
                    and (ptype is None or p["promo_type"] == ptype) and all(lo <= e <= hi for e in p["episodes"])
                    and (not q or q.lower() in (p["hook"] + p["cliffhanger"] + p["why_it_works"] + p["promo_id"]).lower())],
                   key=lambda x: x["post_order"])
    if not shown:
        st.caption("No promos match these filters.")
        return
    if view == "Table":
        st.caption(f"{len(shown)} promos, in posting order")
        st.dataframe(promos_table(shown), hide_index=True, **WIDE,
                     column_config={k: st.column_config.TextColumn(width="large")
                                    for k in ("Hook", "Cliffhanger", "Why it works")})
        return
    pages = max(1, math.ceil(len(shown) / page_size))
    pc = st.columns([1, 6], **VALIGN)
    pg = int(pc[0].number_input("Page", 1, pages, 1, key=f"{key}_page")) if pages > 1 else 1
    pc[1].caption(f"{len(shown)} promos · showing {(pg - 1) * page_size + 1}–{min(len(shown), pg * page_size)} "
                  f"in posting order")
    for p in shown[(pg - 1) * page_size: pg * page_size]:
        promo_card(p, index, durations, f"{key}_{p['promo_id']}", reviewable)


# ===================================================================================== load
ensure_outputs()
D = load_outputs((OUT / "promos_all_episodes.json").stat().st_mtime)
INDEX, _ = corpus(str(config.DEFAULT_INPUT_DIR)) if HAVE_EPISODES else ({}, [])
meta, promos, audit, verify, quality = D["meta"], D["promos"], D["audit"], D["verify"], D["quality"]
DUR = {int(r.episode): float(r.duration_sec) for r in D["stats"].itertuples()}
N_EPS = int(meta["episodes_parsed"])
scorer = meta["provider"].split(":")[-1]
waves = {w: [p for p in sorted(promos, key=lambda x: x["post_order"]) if ui.wave_of(p) == w] for w in (1, 2, 3, 4)}
plan = waves[1] + waves[2] + waves[3]
checks = verify.get("checks", {})
n_pass = sum(1 for c in checks.values() if c.get("pass"))
st.session_state.setdefault("reviews", load_reviews())


# ===================================================================================== header
st.markdown(ui.topbar_html(meta["show_title"], scorer), unsafe_allow_html=True)
st.markdown(ui.hero_html(meta["show_title"], N_EPS, meta["corpus_runtime_min"], len(promos), len(waves[1]),
                         f"{n_pass}/{len(checks)}" if checks else "–", n_plan=len(plan)), unsafe_allow_html=True)
with st.container(key="kpcta"):
    b = st.columns([1.1, 1.1, 1.1, 1.2, 1.5], **VALIGN)
    b[0].download_button("Posting plan (CSV)", (OUT / "ranked_promos.csv").read_bytes(), "ranked_promos.csv",
                         key="dl_csv", **WIDE)
    b[1].download_button("All promos (JSON)", (OUT / "promos_all_episodes.json").read_bytes(), "promos_all_episodes.json",
                         key="dl_json", **WIDE)
    b[2].download_button("Report (Markdown)", (OUT / "ranked_promos_report.md").read_bytes(), "ranked_promos_report.md",
                         key="dl_md", **WIDE)
    if b[3].button("Re-run on all episodes", key="hdr_rerun", disabled=not HAVE_EPISODES,
                   help=None if HAVE_EPISODES else "the episode files are not on this server", **WIDE):
        with st.spinner("Re-running all episodes with the free scorer ..."):
            run_folder(config.DEFAULT_INPUT_DIR, OUT)
        st.cache_data.clear()
        st.rerun()
    if not HAVE_EPISODES:
        b[4].caption("Results-only mode: episode files are not on this server (confidential input pack).")

tab1, tab2, tab3, tab4 = st.tabs([f"Promos ({len(plan)} in the plan)", "What to post first", "Where it picks badly",
                                  "Try a new screenplay"])

# ===================================================================================== 1. promos
with tab1:
    st.markdown(ui.h2("The posting plan",
                      "Each card gives the episode and the in → out timestamp of every segment in the order to cut "
                      "them, then the hook, the cliffhanger and why it works. Approve or reject, nudge the in-point or "
                      "the cut by one screenplay block, then download the approved cut list for the editor."),
                unsafe_allow_html=True)
    rvs = [review_of(p).get("status", "pending") for p in plan]
    r = st.columns([1, 1, 1, 1.6, 1.2], **VALIGN)
    r[0].metric("Approved", rvs.count("approved"))
    r[1].metric("Rejected", rvs.count("rejected"))
    r[2].metric("Not reviewed", rvs.count("pending"))
    r[3].download_button("Approved cut list (CSV)", cut_list_csv(promos, INDEX), "approved_cut_list.csv",
                         key="dl_cuts", disabled=rvs.count("approved") == 0, **WIDE)
    if r[4].button("Clear decisions", key="rv_clear", **WIDE):
        st.session_state["reviews"] = {}
        save_reviews()
        st.rerun()
    show_promos(promos, INDEX, DUR, "t1", N_EPS)

# ===================================================================================== 2. ranking
with tab2:
    W = config.DEFAULT_WEIGHTS
    rk = meta.get("ranking", {})
    R = config.POSTING_RULES
    st.markdown(ui.h2("How promos are ranked",
                      "Six 0–10 scores per promo combine into a score out of 100. Anything that fails one of the "
                      "brief's three tests is held; the rest fill three short lists in score order."),
                unsafe_allow_html=True)
    with st.container(border=True, key="kp_formula"):
        rules = (
            (0, f"Held if: no line or action in the first 3 s, or a hook rated below {R['min_hook']}; an ending that "
                "resolves the scene (last line, final shot or music); makes sense alone below "
                f"{R['min_clarity']}; or a score below {R['min_score']:.0f}."),
            (1, f"The best {R['post_first']} from the first quarter of the show (Ep 1–"
                f"{rk.get('first_quarter_last_episode', '?')}) with spoiler ≤ {R['first_max_spoiler']}, so a click "
                "lands where the story starts. One slot may go to a title reveal from any episode, because the title "
                "already sells it."),
            (2, f"The next {R['post_next']} by score, spoiler ≤ {R['next_max_spoiler']}."),
            (3, f"Up to {R['later']} more; anything with spoiler above {R['next_max_spoiler']} needs a check first."),
            (4, f"No two posts share footage or start where another ends (the later one becomes an alternate for A/B "
                f"tests), and no list holds more than {R['max_per_sequence']} posts from one story sequence."))
        names = {0: ("Tests", ui.CRIMSON), 1: (ui.WAVE_NAMES[1], ui.WAVE_COLORS[1]), 2: (ui.WAVE_NAMES[2], ui.WAVE_COLORS[2]),
                 3: (ui.WAVE_NAMES[3], ui.WAVE_COLORS[3]), 4: ("No repeats", ui.WAVE_COLORS[4])}
        st.markdown(
            '<div class="kp-line"><span class="k" style="color:#FF5A36">Score</span><span class="t">'
            + " + ".join(f"{abs(v) * 100:.0f}% {ui.SCORE_LABELS[k].lower()}" for k, v in W.items() if v > 0)
            + f" − {abs(W['spoiler_risk']) * 100:.0f}% spoiler risk, scaled to 100</span></div>"
            + "".join(f'<div class="kp-line"><span class="k" style="color:{names[i][1]}">{names[i][0]}</span>'
                      f'<span class="t">{ui.esc(t)}</span></div>' for i, t in rules),
            unsafe_allow_html=True)
        st.caption(f"This run held {rk.get('held_no_hook', 0)} for the hook, {rk.get('held_resolved', 0)} for a resolved "
                   f"ending, {rk.get('held_unclear', 0)} as unclear and {rk.get('held_low_score', 0)} for a low score; "
                   f"{rk.get('alternates', 0)} are alternates of a posted promo.")
    for w, title in ((1, "Post these first"), (2, "Post next"), (3, "Later")):
        st.markdown(ui.h2(f"{title} · {len(waves[w])} promos"), unsafe_allow_html=True)
        with st.container(border=True, key=f"kp_wave{w}"):
            st.markdown("".join(ui.rank_row_html(p) for p in waves[w]) or "None in this run.", unsafe_allow_html=True)
    st.caption(f"{len(waves[4])} promos are held (reasons in held_promos.csv and on each card under Show → Held).")

# ===================================================================================== 3. quality
with tab3:
    st.markdown(ui.h2("Where the tool picks badly, and how we know",
                      "A blind second judge on every post in the plan, the same judge on earlier plans, repetition, "
                      "and how the plan compares with published teardowns of real short-drama ads."), unsafe_allow_html=True)
    jd, rep_, ads = quality.get("second_judge", {}), quality.get("repetition", {}), quality.get("real_ads", {})
    PLAN_NAMES = {"before_gemini:wave1": "Before fixes: Gemini run, Wave 1",
                  "before_heuristic:wave1": "Before fixes: keyword run, Wave 1"}

    def _jrow(name, m):
        if not m.get("judged"):
            return {"List": name, "Posts": m.get("posts", 0), "Judged": 0}
        return {"List": name, "Posts": m["posts"], "Judged": m["judged"], "Post score": m["post_score"],
                "All three tests": f"{m['all_three_tests']}/{m['judged']}",
                "Soft or closed endings": m["weak_or_closed_endings"], "Weak hooks": m["weak_hooks"]}
    def _judge_words(s):
        d = dict(x.strip().split(" ") for x in s.split(","))
        words = {"hook": {"S": "strong hook", "W": "weak hook", "F": "no hook"},
                 "ending": {"O": "open ending", "W": "soft ending", "R": "resolved ending"},
                 "alone": {"Y": "makes sense alone", "P": "partly clear alone", "N": "unclear alone"},
                 "post": {"Y": "would post", "M": "maybe", "N": "would not post"}}
        return ", ".join(words[k].get(v, v) for k, v in d.items() if k in words)
    with st.container(border=True, key="kp_q_judge"):
        st.markdown('<div class="kp-h2">Blind second judge</div>', unsafe_allow_html=True)
        if jd.get("available"):
            rows = [_jrow(f"This run: {k}", m) for k, m in jd["this_run"].items() if k != "Later"]
            rows.append(_jrow("This run: labelled after the main fixes", jd.get("this_run_labelled_after_main_fixes", {})))
            rows += [_jrow(PLAN_NAMES.get(k, k), m) for k, m in jd.get("earlier_plans", {}).items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, **WIDE)
            h = jd.get("held_for_failing_a_test", {})
            st.caption(f"Post score: would post = 1, maybe = 0.5, no = 0. Of {h.get('judged', 0)} promos the rules held "
                       f"for failing a test, the judge would not post {h.get('judge_would_not_post', 0)}, says maybe to "
                       f"{h.get('judge_maybe', 0)} and would post {h.get('judge_would_post', 0)}. " + jd["provenance"])
        else:
            st.caption(jd.get("note", "No judge labels found (labels/judge_*.csv)."))
    b1, b2 = st.columns(2, gap="medium")
    with b1:
        with st.container(border=True, key="kp_q_bad"):
            st.markdown('<div class="kp-h2">Bad picks the judge found in this plan</div>', unsafe_allow_html=True)
            bad = jd.get("plan_promos_the_judge_faults", [])
            st.markdown("\n".join(f"- **{b['list']} #{b['post_order']}** {b['clip']}: {_judge_words(b['judge'])}"
                                   for b in bad) or "None.")
            held_y = jd.get("held_promos_the_judge_would_post", [])
            if held_y:
                st.markdown("**Held, but the judge would post them** (rules too strict):\n" + "\n".join(
                    f"- {b['clip']}: {b['held_because']}" for b in held_y))
    with b2:
        with st.container(border=True, key="kp_q_rep"):
            st.markdown('<div class="kp-h2">Repetition</div>', unsafe_allow_html=True)
            if rep_.get("available"):
                rr = [(k, m) for k, m in rep_["this_run"].items()] + \
                     [(PLAN_NAMES.get(k, k), m) for k, m in rep_.get("earlier_plans", {}).items()]
                st.markdown("| List | Posts | Episodes | Max per story sequence |\n|---|---|---|---|\n" + "\n".join(
                    f"| {k} | {m['posts']} | {m['episodes']} | {m['most_from_one_sequence']} |" for k, m in rr))
                big = max(rr, key=lambda km: km[1]["most_from_one_sequence"])[1].get("that_sequence", "")
                eps = rep_.get("sequence_episodes", {}).get(big, [])
                st.caption("A story sequence is one stretch of story in one place"
                           + (f" (the busiest here runs Ep {eps[0]}-{eps[-1]})" if eps else "")
                           + ". Older plans posted one sequence again and again; now each list allows two.")
    with st.container(border=True, key="kp_q_ads"):
        st.markdown('<div class="kp-h2">The plan next to real ReelShort and DramaBox ads</div>', unsafe_allow_html=True)
        if ads.get("plan"):
            st.markdown("| Pattern | Real ads | This plan |\n|---|---|---|\n" + "\n".join(
                f"| **{r['pattern']}** | {r['real_ads']} | {r['this_plan']} |" for r in ads["rows"]))
            st.caption(ads["caveat"] + " Sources: " + " · ".join(f"[{s['title']}]({s['url']})"
                                                                 for s in ads["sources"].values()))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True, key="kp_q_rules"):
            st.markdown('<div class="kp-h2">Rule checks</div>', unsafe_allow_html=True)
            st.markdown("\n".join(f"- **{g['rule']}**: {g['effect']}" for g in quality.get("rule_checks", [])))
            st.caption("These show the rules ran on this show, not that the picks are good; the judge above is the "
                       "outside check.")
    with c2:
        with st.container(border=True, key="kp_q_other"):
            st.markdown('<div class="kp-h2">Other scorers and human labels</div>', unsafe_allow_html=True)
            ag, hl = quality.get("scorer_agreement", {}), quality.get("human_labels", {})
            if ag.get("available"):
                st.markdown(f"Model vs keyword scorer: rank correlation **{ag['composite_spearman']}**, top-10 overlap "
                            f"**{ag['top10_overlap']}/10**.")
            for cmp in quality.get("comparisons", []):
                t15 = cmp.get("judge_other_top15", {})
                st.markdown(f"vs **{cmp['scorer']}**{' (cache-only preview)' if cmp.get('cache_only_preview') else ''}: "
                            f"top-15 overlap {cmp['top15_overlap']}/15; the judge gives its top 15 "
                            f"{t15.get('post_score', '-')} ({t15.get('judged', 0)} judged)")
            if hl.get("available"):
                st.markdown(f"People: precision@10 **{hl['precision_at_10']}**, plan precision **{hl['plan_precision']}**, "
                            f"kappa tool-vs-human **{hl['kappa_tool_vs_human']}**")
            else:
                st.caption("Human labels: " + hl.get("note", "") + " (labels/README.md)")
    cat = {"where_ai_got_it_wrong": ("AI got it wrong", ui.CRIMSON), "guardrail_caught": ("Caught by a guardrail", "#10B981"),
           "where_tool_picks_badly": ("Still picks badly", ui.AMBER), "tool_limit_fixed": ("Tool limit, fixed", "#3B82F6"),
           "not_applicable": ("Not in this input", "#64748B")}
    st.markdown(ui.h2("Failure modes found on this show"), unsafe_allow_html=True)
    for fm in audit.get("failure_modes", []):
        lab, col = cat.get(fm["category"], (fm["category"], ui.TEXT_3))
        with st.container(border=True, key=f"kp_fm_{fm['case_id']}"):
            st.markdown(
                f'<div class="kp-card-h">{ui.badge(lab, col)}<span style="font-weight:700;color:{ui.TEXT_1}">'
                f'{ui.esc(fm["failure_mode"])}</span></div>'
                f'<div class="kp-note" style="margin-bottom:8px">Example: {ui.esc(fm["title"])}</div>'
                f'<div class="kp-line"><span class="k" style="color:#34D399">Fix</span><span class="t">{ui.esc(fm["fix"])}</span></div>'
                f'<div class="kp-line"><span class="k" style="color:{ui.AMBER}">Still risky</span><span class="t">'
                f'{ui.esc(fm["residual_risk"])}</span></div>', unsafe_allow_html=True)
            with st.expander("Before / after and screenplay evidence"):
                st.json({"without the guard": fm.get("naive_pick"), "with the guard": fm.get("guarded_pick")})
                if fm.get("evidence"):
                    st.code("\n".join(fm["evidence"]), language=None)
    flagged = audit.get("flagged_ranked_picks", [])
    if flagged:
        st.markdown(ui.h2(f"Plan promos flagged for a human look · {len(flagged)}"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{
            "Promo": f["promo_id"], "Wave": ui.WAVE_NAMES[ui.wave_of(f)], "Score": f["composite"],
            "Segments": "  +  ".join(f"Ep {s['episode']} {s['start_timestamp']}→{s['end_timestamp']}" for s in f["segments"]),
            "Why flagged": "; ".join(f["why_flagged"])} for f in flagged]), hide_index=True, **WIDE)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(ui.h2(f"Automatic checks · {n_pass}/{len(checks)} passed"), unsafe_allow_html=True)
        with st.container(border=True, key="kp_checks"):
            st.markdown("".join(ui.check_row(c["pass"], k, c["detail"]) for k, c in checks.items())
                        or "No verification report found.", unsafe_allow_html=True)
    with c2:
        st.markdown(ui.h2("Human check before posting", "About 10 minutes per show."), unsafe_allow_html=True)
        with st.container(border=True, key="kp_hitl"):
            st.markdown(ui.steps_html(audit.get("human_review_protocol", [])), unsafe_allow_html=True)

# ===================================================================================== 4. new input
with tab4:
    st.markdown(ui.h2("Run it on a new screenplay",
                      "Same pipeline as the command line. Upload or paste episodes in the same screenplay format (or "
                      "pick episodes from this show), choose the re-ranker for the shortlist, and run. The cast, the "
                      "lead and the title character are learned from the new screenplays."), unsafe_allow_html=True)
    with st.container(border=True, key="kp_runner"):
        modes = ["Upload .txt screenplay(s)", "Paste screenplay text"] + (["Episodes from this show"] if HAVE_EPISODES else [])
        mode = st.radio("Input", modes, horizontal=True, key="live_mode")
        texts: List[str] = []
        names: List[str] = []
        chosen: List[int] = []
        if mode == "Episodes from this show":
            chosen = st.multiselect("Episodes", sorted(INDEX), default=sorted(INDEX)[:3], key="live_eps")
        elif mode == "Upload .txt screenplay(s)":
            for up in st.file_uploader("Screenplay .txt files", type=["txt"], accept_multiple_files=True, key="live_up") or []:
                texts.append(up.read().decode("utf-8", errors="replace"))
                names.append(up.name)
        else:
            txt = st.text_area("Paste one episode", height=200, key="live_txt",
                               placeholder="EPISODE NUMBER ... --- SCRIPT START --- ... Start: (MM:SS.ss) | End: (MM:SS.ss)")
            if txt.strip():
                texts, names = [txt], ["pasted-episode.txt"]
        show_title = st.text_input("Show title (used to recognise premise reveals)", value="" if mode != "Episodes from this show"
                                   else meta["show_title"], key="live_title")
        c = st.columns(3)
        prov_label = c[0].selectbox("Stage 2: re-ranker for the shortlist", list(PROVIDERS), key="live_prov",
                                    help="Stage 1 (code) cuts and scores every candidate and keeps the best "
                                         f"{config.SHORTLIST_SIZE}. Stage 2 orders that shortlist and writes the "
                                         "why-it-works line: keyword is free and instant; a model answer is cached, "
                                         "so a repeat run is instant too.")
        provider = PROVIDERS[prov_label]
        model = c[1].text_input("Model id", value=PROVIDER_MODEL.get(prov_label, config.PROVIDER_DEFAULT_MODELS[provider]),
                                disabled=provider == "heuristic", key=f"live_model_{prov_label}")
        api_key = c[2].text_input("API key (optional, else read from .env)", type="password",
                                  disabled=provider == "heuristic", key="live_key")
        with st.expander("Advanced settings"):
            dur = st.slider("Promo length (s)", 10.0, 90.0, (config.MIN_DURATION, config.MAX_DURATION), step=1.0,
                            key="live_dur")
            wc = st.columns(6)
            weights = {k: wc[i].number_input(ui.SCORE_LABELS[k], -1.0, 1.0, float(v), 0.01, key=f"w_{k}")
                       for i, (k, v) in enumerate(config.DEFAULT_WEIGHTS.items())}
            bc = st.columns(2)
            budget = bc[0].number_input("Model time budget (s); later groups keep the keyword order", 10.0, 900.0,
                                        config.RERANK_TIME_BUDGET_SEC, 10.0, key="live_budget",
                                        disabled=provider == "heuristic")
            pacing = bc[1].number_input("Seconds between model calls", 0.0, 60.0, config.LLM_PACING_SEC, 1.0,
                                        key="live_pacing")
        run_clicked = st.button("Find promos", type="primary", key="live_run")
    if run_clicked:
        log = st.empty()
        msgs: List[str] = []

        def say(m: str) -> None:
            msgs.append(m)
            log.code("\n".join(msgs[-8:]), language=None)
        kw = dict(provider=provider, model=model if provider != "heuristic" else None, api_key=api_key or None,
                  min_duration=dur[0], max_duration=dur[1], weights=weights, time_budget=budget,
                  pacing_sec=pacing, progress=say)
        res = None
        try:
            if mode == "Episodes from this show":
                eps = [INDEX[e].ep for e in chosen]
                res = run_on_episodes(eps, show_title or meta["show_title"], **kw) if eps else None
            else:
                res = run_texts(texts, names, show_title=show_title or "Uploaded screenplay", **kw) if texts else None
            if res is None:
                st.warning("Upload, paste or pick at least one episode first.")
        except Exception as exc:   # parse / model errors are shown, never swallowed
            st.error(f"Run failed: {exc!r}")
        if res is not None:
            st.session_state["live"] = {
                "promos": [x.to_dict() for x in sorted(res.promos, key=lambda z: z.post_order)], "meta": res.meta,
                "index": {e: res.generator.index[e] for e in res.generator.index},
                "durations": {e: res.generator.index[e].ep.duration_sec for e in res.generator.index}}
    live = st.session_state.get("live")
    if live:
        lm = live["meta"]
        lw = {w: sum(1 for p in live["promos"] if ui.wave_of(p) == w) for w in (1, 2, 3, 4)}
        prof = lm.get("show_profile", {})
        usage = lm.get("llm_usage")
        cost = f" · {usage['calls']} model calls · ${usage['cost_usd']:.4f}" if usage else ""
        st.success(f"{lm['promos']} promos from {lm['episodes_parsed']} episode(s) in {lm['latency_sec']['total']:.1f} s "
                   f"· post first {lw[1]} · next {lw[2]} · later {lw[3]} · held {lw[4]} · {lm['shortlisted']} "
                   f"shortlisted, re-ranked by {lm['reranker']}{cost}")
        sb = lm.get("show_brief", {})
        if sb:
            st.caption(f"Premise learned from the title and the script: {sb.get('premise_setup', '')} → "
                       f"{sb.get('premise_twist', '')}. Stakes words: {', '.join(sb.get('stakes_terms', [])[:8]) or '-'}.")
        if prof:
            st.caption(f"Learned from the screenplay: lead {', '.join(prof.get('protagonists', [])) or '-'}; title "
                       f"character {', '.join(sorted(set(prof.get('premise_cast', [])) - set(prof.get('protagonists', [])))) or '-'}; "
                       f"{len(prof.get('core_cast', []))} recurring characters.")
        max_live_ep = max((e for p in live["promos"] for e in p["episodes"]), default=1)
        show_promos(live["promos"], live["index"], live.get("durations", DUR), "res", max_live_ep, page_size=8,
                    reviewable=False)

