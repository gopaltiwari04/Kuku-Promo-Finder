"""Quality evaluation of a finished run: how good the picks are, measured from outside the rules where possible.

Writes raw_outputs/quality_eval.json (+ .md). No API calls. Sections, most independent first:

  1. Second judge       blind labels in labels/judge_*.csv (hook / ending / makes sense alone / would post) matched to
                        this run's posting plan, the plans of earlier runs, and the promos the rules held back
  2. Repetition         posts per story sequence, for this plan and for earlier plans
  3. Other runs         stage-2 model re-rank vs the code score; --compare <output dir> for another run
  4. Real ads           the plan next to what published teardowns of ReelShort / DramaBox ads report
  5. Rule checks        what each rule changed on this show (these show the rules ran, not that the picks are good)
  6. Human labels       once labels/label_sheet.csv is filled in by people
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import config
from .lexicons import TROPE_LABELS

DIMS = ["hook_strength", "cliffhanger_tension", "standalone_clarity", "trope_virality", "spoiler_risk"]
LISTS = {1: "Post first", 2: "Post next", 3: "Later"}
POST_VALUE = {"Y": 1.0, "M": 0.5, "N": 0.0}
JUDGE_PROVENANCE = (
    "Blind second judge: an AI model reading each clip as screenplay lines only (no episode number, score, rank, "
    "list or tool copy), in shuffled packets. Labels: hook S strong / W weak / F fails; ending O open / W soft / "
    "R resolved; makes sense alone Y / P partly / N; would post Y / M maybe / N. It is not a person and not "
    "independent of the tools used to build this, so it is a second opinion, not ground truth. Rounds 1-3 were labelled while the "
    "rules were being fixed (the bad picks they found drove those fixes). Round 4 was labelled after those fixes; its "
    "weak hooks led to one last fix (a line that merely starts with 'You', a name on its own, or a mention of guards "
    "no longer counts as a hook). Round 5 labelled the posts that fix brought in.")


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return round(cov / (va * vb), 3) if va and vb else None


def kappa(a: Sequence[int], b: Sequence[int]) -> Optional[float]:
    n = len(a)
    if n < 5:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def _composite(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    return max(0.0, min(100.0, 10 * sum(weights[k] * float(scores.get(k, 0)) for k in weights)))


def _load(out_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    data = json.loads((out_dir / "promos_all_episodes.json").read_text(encoding="utf-8"))
    meta = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    audit = json.loads((out_dir / "quality_audit_and_bad_picks.json").read_text(encoding="utf-8"))
    return data["promos"], meta, audit


def _ts(t: str) -> float:
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def clip_key(p: Dict[str, Any]) -> str:
    """'Ep12 00:26.80-01:09.20 + Ep13 ...': the label files identify clips by timings only (no screenplay text)."""
    return " + ".join(f"Ep{s['episode']} {s['start_timestamp']}-{s['end_timestamp']}" for s in p["segments"])


def plan_lists(promos: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    plan = sorted([p for p in promos if p.get("wave") in LISTS], key=lambda p: p["post_order"])
    return {"Post first": [p for p in plan if p["wave"] == 1], "Top 15": plan[:15],
            "Later": [p for p in plan if p["wave"] == 3], "Whole plan": plan}


def held_for_quality(p: Dict[str, Any]) -> bool:
    """Held because a test failed (hook, ending, clarity, score), not because the plan was full or it repeats a post."""
    r = p.get("wave_reason", "")
    return p.get("wave") == 4 and r.startswith("held:") and "plan is full" not in r


def _before_plans(J: Dict[str, Dict[str, str]]) -> Dict[str, List[Tuple[int, Dict[str, str]]]]:
    """Earlier plans as recorded in the label file: tag 'before_gemini:wave1#3' = 3rd post of that run's first list."""
    out: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for r in J.values():
        for tag in filter(None, (t.strip() for t in r.get("before_plans", "").split(";"))):
            plan_name, pos = tag.split("#")
            out.setdefault(plan_name, []).append((int(pos), r))
    return {k: sorted(v, key=lambda t: t[0]) for k, v in sorted(out.items()) if k.endswith("wave1")}


# ------------------------------------------------------------------------------------------- 1. second judge
def load_judge(labels_dir: Path) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
    files = sorted(labels_dir.glob("judge_*.csv")) if labels_dir.exists() else []
    if not files:
        return {}, None
    with open(files[0], encoding="utf-8") as fh:
        return {r["clip"]: r for r in csv.DictReader(fh)}, files[0].name


def judge_metrics(labels: Iterable[Dict[str, str]], n: Optional[int] = None) -> Dict[str, Any]:
    L = list(labels)
    if not L:
        return {"posts": n or 0, "judged": 0}
    return {"posts": n if n is not None else len(L), "judged": len(L),
            "post_score": round(sum(POST_VALUE[r["would_post"]] for r in L) / len(L), 2),
            "all_three_tests": sum(r["hook"] == "S" and r["ending"] == "O" and r["makes_sense_alone"] == "Y" for r in L),
            "weak_or_closed_endings": sum(r["ending"] in ("W", "R") for r in L),
            "weak_hooks": sum(r["hook"] in ("W", "F") for r in L),
            "would_not_post": sum(r["would_post"] == "N" for r in L)}


def _labels_str(r: Dict[str, str]) -> str:
    return f"hook {r['hook']}, ending {r['ending']}, alone {r['makes_sense_alone']}, post {r['would_post']}"


def second_judge(promos: List[Dict[str, Any]], labels_dir: Path) -> Dict[str, Any]:
    J, name = load_judge(labels_dir)
    if not J:
        return {"available": False, "note": "no labels/judge_*.csv found"}
    lists = plan_lists(promos)
    out: Dict[str, Any] = {"available": True, "file": name, "provenance": JUDGE_PROVENANCE, "clips_labelled": len(J),
                           "this_run": {k: judge_metrics([J[clip_key(p)] for p in v if clip_key(p) in J], len(v))
                                        for k, v in lists.items()}}
    plan = lists["Whole plan"]
    fresh = [J[clip_key(p)] for p in plan if clip_key(p) in J and J[clip_key(p)].get("round") in ("4", "5")]
    out["this_run_labelled_after_main_fixes"] = judge_metrics(fresh, len(fresh))
    out["earlier_plans"] = {k: judge_metrics([r for _, r in v]) for k, v in _before_plans(J).items()}
    qh = [p for p in promos if held_for_quality(p) and clip_key(p) in J]
    out["held_for_failing_a_test"] = {"judged": len(qh),
                                      "judge_would_not_post": sum(J[clip_key(p)]["would_post"] == "N" for p in qh),
                                      "judge_maybe": sum(J[clip_key(p)]["would_post"] == "M" for p in qh),
                                      "judge_would_post": sum(J[clip_key(p)]["would_post"] == "Y" for p in qh)}
    bad = []
    for p in plan:
        r = J.get(clip_key(p))
        if r and (r["would_post"] == "N" or r["hook"] in ("W", "F") or r["ending"] in ("W", "R")):
            bad.append({"promo_id": p["promo_id"], "list": LISTS[p["wave"]], "post_order": p["post_order"],
                        "clip": clip_key(p), "judge": _labels_str(r)})
    out["plan_promos_the_judge_faults"] = bad
    out["held_promos_the_judge_would_post"] = [
        {"promo_id": p["promo_id"], "clip": clip_key(p), "held_because": p["wave_reason"].replace("held: ", ""),
         "judge": _labels_str(J[clip_key(p)])} for p in qh if J[clip_key(p)]["would_post"] == "Y"]
    out["unjudged_plan_promos"] = [p["promo_id"] for p in plan if clip_key(p) not in J]
    return out


# ------------------------------------------------------------------------------------------- 2. repetition
def _parse_clip(clip: str) -> List[Tuple[int, float, float]]:
    out = []
    for part in clip.split(" + "):
        m = re.match(r"Ep(\d+) (\d+:\d+\.\d+)-(\d+:\d+\.\d+)", part.strip())
        if m:
            out.append((int(m.group(1)), _ts(m.group(2)), _ts(m.group(3))))
    return out


def _sequences_of(segments: Sequence[Tuple[int, float, float]], spans: Dict[str, List[List[float]]]) -> List[str]:
    return [sid for sid, sp in spans.items()
            if any(ep == e and min(b, t1) - max(a, t0) > 0.5 for ep, t0, t1 in segments for e, a, b in sp)]


def repetition_of(clips: List[str], spans: Dict[str, List[List[float]]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    eps = set()
    for c in clips:
        segs = _parse_clip(c)
        eps |= {e for e, _, _ in segs}
        for sid in _sequences_of(segs, spans):
            counts[sid] = counts.get(sid, 0) + 1
    top = max(counts.items(), key=lambda t: t[1]) if counts else ("-", 0)
    return {"posts": len(clips), "episodes": len(eps), "story_sequences": len(counts),
            "most_from_one_sequence": top[1], "that_sequence": top[0]}


def repetition(promos: List[Dict[str, Any]], meta: Dict[str, Any], labels_dir: Path) -> Dict[str, Any]:
    spans = meta.get("story_sequences") or {}
    if not spans:
        return {"available": False}
    J, _ = load_judge(labels_dir)
    out: Dict[str, Any] = {
        "available": True,
        "this_run": {k: repetition_of([clip_key(p) for p in v], spans) for k, v in plan_lists(promos).items()},
        "earlier_plans": {k: repetition_of([r["clip"] for _, r in v], spans) for k, v in _before_plans(J).items()},
        "sequence_episodes": {sid: sorted({int(e) for e, _, _ in sp}) for sid, sp in spans.items()}}
    return out


# ------------------------------------------------------------------------------------------- 3. other runs
def scorer_agreement(promos: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    """Stage 2 vs stage 1 on the shortlist: does the model re-ranker order the clips like the code score does?"""
    live = [p for p in promos if (p.get("signals") or {}).get("shortlisted")
            and (p.get("signals") or {}).get("reranked_by") not in (None, "keyword")
            and (p.get("signals") or {}).get("rerank_score") is not None]
    if len(live) < 5:
        return {"available": False, "note": "stage 2 was the free keyword re-ranker, so the order is the code score; "
                                            "a model re-rank adds a side-by-side here"}
    rr = [float(p["signals"]["rerank_score"]) for p in live]
    code = [float(p["signals"]["code_score"]) for p in live]
    top_m = {p["promo_id"] for p, _ in sorted(zip(live, rr), key=lambda t: -t[1])[:10]}
    top_c = {p["promo_id"] for p, _ in sorted(zip(live, code), key=lambda t: -t[1])[:10]}
    return {"available": True, "promos": len(live), "reranked_by": sorted({p["signals"]["reranked_by"] for p in live}),
            "composite_spearman": spearman(rr, code), "top10_overlap": len(top_m & top_c)}


def compare_runs(promos: List[Dict[str, Any]], other: List[Dict[str, Any]], label: str,
                 judge: Dict[str, Dict[str, str]], other_meta: Dict[str, Any]) -> Dict[str, Any]:
    a = {clip_key(p): p for p in promos}
    b = {clip_key(p): p for p in other}
    shared = sorted(set(a) & set(b))
    in_plan = lambda p: p.get("wave") in LISTS
    la, lb = plan_lists(promos), plan_lists(other)
    keys = lambda ps: [clip_key(p) for p in ps]
    res: Dict[str, Any] = {"label": label, "scorer": other_meta.get("provider"), "shared_clips": len(shared),
                           "cache_only_preview": bool(other_meta.get("cache_only_preview")),
                           "post_first_overlap": len(set(keys(la["Post first"])) & set(keys(lb["Post first"]))),
                           "top15_overlap": len(set(keys(la["Top 15"])) & set(keys(lb["Top 15"])))}
    if len(shared) >= 5:
        res["score_spearman"] = spearman([a[k]["composite_score"] for k in shared], [b[k]["composite_score"] for k in shared])
        res["same_plan_decision_pct"] = round(100 * sum(in_plan(a[k]) == in_plan(b[k]) for k in shared) / len(shared), 1)
    if judge:
        res["judge_other_post_first"] = judge_metrics([judge[k] for k in keys(lb["Post first"]) if k in judge],
                                                      len(lb["Post first"]))
        res["judge_other_top15"] = judge_metrics([judge[k] for k in keys(lb["Top 15"]) if k in judge], len(lb["Top 15"]))
    return res


# ------------------------------------------------------------------------------------------- 4. real ads
# Published teardowns of real short-drama ads: third-party estimates and contact sheets; we did not watch the ads.
REAL_AD_SOURCES = {
    "admapix_reelshort": ("AdMapix, ReelShort ads 2026 (US, 4 Oct-2 Nov 2026, 8 contact sheets at 0/2/5/10 s)",
                          "https://www.admapix.com/blog/ad-intelligence/reelshort-ad-strategy-2026"),
    "admapix_dramabox": ("AdMapix, DramaBox ad strategy 2026 (US, 1 Jul-14 Aug 2026, top 10 by modelled impressions)",
                         "https://www.admapix.com/blog/ad-intelligence/dramabox-ad-strategy-2026"),
    "invideo": ("invideo, micro-drama hooks and cliffhangers", "https://invideo.io/blog/ai-micro-drama-hooks-cliffhangers/"),
    "thewrap": ("TheWrap, interview with DramaShorts co-founder Leo Ovdiienko",
                "https://www.thewrap.com/dramashorts-co-founder-leo-ovdiienko-guilty-pleasure-viewing-interview"),
    "socialpeta": ("SocialPeta, short-drama weekly ranking, 16-22 Feb 2026",
                   "https://socialpeta.com/en/blog/short-drama-weekly-rankin-how-phased-ad-strategies-unlock-viral-growth"),
}


def _first_beat_offset(p: Dict[str, Any]) -> Optional[float]:
    m = re.match(r"^\[(?:Ep\d+ )?(\d{2}):(\d{2}\.\d{2})\]", p.get("hook", ""))
    if not m:
        return None
    return round(int(m.group(1)) * 60 + float(m.group(2)) - _ts(p["segments"][0]["start_timestamp"]), 2)


def real_ads(promos: List[Dict[str, Any]], judge_part: Dict[str, Any]) -> Dict[str, Any]:
    lists = plan_lists(promos)
    plan, first = lists["Whole plan"], lists["Post first"]
    if not plan:
        return {"plan": 0}
    offs = [o for o in (_first_beat_offset(p) for p in plan) if o is not None]
    kinetic = sum(p.get("signals", {}).get("hook_type") == "kinetic" for p in plan)
    durs = sorted(p["total_duration_sec"] for p in plan)
    multi = sum(len(p["segments"]) > 1 for p in plan)
    alts = sum(1 for p in promos if p.get("variant_of"))
    tropes = []
    for p in first:
        tags = p.get("arc_tags") or p.get("signals", {}).get("tags") or []
        tropes.append(TROPE_LABELS.get(tags[0], tags[0]).lower() if tags else "-")
    jw = (judge_part.get("this_run", {}).get("Whole plan") or {}) if judge_part.get("available") else {}
    last_ep = max((s["episode"] for p in first for s in p["segments"]), default=0)
    rows = [
        {"pattern": "Open on a line or a visible threat", "source": "admapix_reelshort",
         "real_ads": "7 of 8 sampled ReelShort ads show a relationship or conflict line, a visible threat or a genre cue "
                     "by 2 s (e.g. a divorce line at 0 s); the 8th is a catalogue montage",
         "this_plan": f"{sum(o <= 1.0 for o in offs)} of {len(plan)} posts have their first line or action within 1 s "
                      f"and all within 3 s ({len(plan) - kinetic} open on a line, {kinetic} on physical action)"},
        {"pattern": "Length", "source": "admapix_reelshort; admapix_dramabox",
         "real_ads": "ReelShort sample 12-59 s (median 21.5 s). DramaBox top 10: 7 entries run 46-64 s; 3 run 111-116 s "
                     "and hold 77.8% of the modelled impressions",
         "this_plan": f"{durs[0]:.0f}-{durs[-1]:.0f} s, median {durs[len(durs) // 2]:.0f} s ({multi} joined cuts); the "
                      f"brief caps promos at 60 s, so the 2-minute format DramaBox spends most on is out of scope"},
        {"pattern": "Cut before the resolution", "source": "invideo; admapix_reelshort",
         "real_ads": "end one beat before the resolution; where the ReelShort sheets show an ending, the choice or the "
                     "dispute is left unresolved",
         "this_plan": "every post passes the open-ending rule"
                      + (f"; the second judge calls {jw.get('weak_or_closed_endings', 0)} of {jw.get('judged', 0)} "
                         f"endings soft or closed" if jw.get("judged") else "")},
        {"pattern": "Story type", "source": "admapix_dramabox; admapix_reelshort; socialpeta",
         "real_ads": "power reversal, dangerous protector, relationship contract (DramaBox); wedding encounters, "
                     "ultimatums, physical danger (ReelShort); hidden identity, public ridicule, revenge (SocialPeta)",
         "this_plan": "Post first leads with " + ", ".join(tropes)},
        {"pattern": "Where the click lands", "source": "thewrap",
         "real_ads": "take the hook and the most provocative parts, then lead viewers into the app",
         "this_plan": f"Post first comes from the first quarter of the show (Ep 1-{last_ep}), so a click starts near "
                      f"episode one"},
        {"pattern": "Many versions of one video", "source": "admapix_reelshort",
         "real_ads": "25 ad records share 14 video files",
         "this_plan": f"{alts} alternate cuts kept next to the plan for A/B tests"},
    ]
    return {"plan": len(plan), "rows": rows,
            "sources": {k: {"title": t, "url": u} for k, (t, u) in REAL_AD_SOURCES.items()},
            "caveat": "Third-party modelled estimates and contact sheets. They describe what top ads look like, not "
                      "what makes them convert, and none of them covers this show."}


# ------------------------------------------------------------------------------------------- 5. rule checks
def rule_checks(promos: List[Dict[str, Any]], meta: Dict[str, Any], audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    g = meta.get("guardrail_stats", {})
    rk = meta.get("ranking", {})
    fm = {c["case_id"]: c for c in audit.get("failure_modes", [])}
    rows = [
        {"rule": "3-second hook gate", "effect": f"{meta.get('clean_starts', 0) - meta.get('starts_passing_hook_gate', 0):,} "
         f"of {meta.get('clean_starts', 0):,} possible start points rejected (no line or physical action in the first 3 s)"},
        {"rule": "ending check (last line + final shots and sound)", "effect": f"{g.get('closing_endings', '?')} of "
         f"{g.get('clean_ends', '?')} possible cut points resolve the scene and are marked down; "
         f"{rk.get('held_resolved', 0)} promo(s) held for a resolved ending"},
        {"rule": "same-villain check on joined cuts",
         "effect": f"{(fm.get('FM6', {}).get('metrics') or {}).get('blocked_arcs', '?')} cross-episode cuts blocked "
                   "because the payback shows a different villain; 0 shipped"},
        {"rule": "blind scoring", "effect": fm.get("FM3", {}).get("title", "headers never reach the scorer")},
        {"rule": "repeat control", "effect": f"{rk.get('alternates', 0)} promos kept as alternates (shared footage or "
         f"a start where another post ends); at most {config.POSTING_RULES['max_per_sequence']} posts per story "
         f"sequence in each list"},
        {"rule": "hold rules", "effect": f"{rk.get('held_no_hook', 0)} held for the hook, {rk.get('held_resolved', 0)} "
         f"for the ending, {rk.get('held_unclear', 0)} as unclear, {rk.get('held_low_score', 0)} for a low score"},
    ]
    rows.append({"rule": "tight starts", "effect": f"{meta.get('starts_tightened', 0)} segment starts trimmed to "
                 f"{meta.get('pre_roll_sec', 0.3)} s before the first line or action"})
    rows.append({"rule": "premise slot", "effect": (f"{rk['premise_pick']} reserved in Post first (sells the title's "
                 "twist)") if rk.get("premise_pick") else "no promo sells the title's twist and passes the tests"})
    if rk.get("finale_first_episode"):
        rows.append({"rule": "finale out of cold lists", "effect": f"Ep {rk['finale_first_episode']} onward is the "
                     f"finale; {rk.get('finale_to_later', 0)} finale promo(s) moved to Later as retarget-only"})
    return rows


# ------------------------------------------------------------------------------------------- 6. human labels
YES = {"y", "yes", "1", "true", "x"}
NO = {"n", "no", "0", "false"}


def human_labels(promos: List[Dict[str, Any]], labels_dir: Path) -> Dict[str, Any]:
    sheet, key = labels_dir / "label_sheet.csv", labels_dir / "label_key.csv"
    if not (sheet.exists() and key.exists()):
        return {"available": False, "note": "run scripts/make_label_sheet.py and fill in labels/label_sheet.csv"}
    with open(key, encoding="utf-8") as fh:
        k = {r["item"]: r["promo_id"] for r in csv.DictReader(fh)}
    with open(sheet, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {p["promo_id"]: p for p in promos}
    top10_ids = {p["promo_id"] for p in plan_lists(promos)["Whole plan"][:10]}
    r1, r2, model, top10 = [], [], [], []
    for r in rows:
        pid = k.get(r["item"])
        if pid not in by_id:
            continue
        a, b = r.get("rater1_would_post", "").strip().lower(), r.get("rater2_would_post", "").strip().lower()
        if a not in YES | NO:
            continue
        av = int(a in YES)
        bv = int(b in YES) if b in YES | NO else None
        r1.append(av)
        r2.append(bv)
        model.append(int(by_id[pid].get("wave") in LISTS))
        if pid in top10_ids:
            top10.append(av if bv is None else int(av and bv))
    if len(r1) < 10:
        return {"available": False, "note": f"only {len(r1)} rows labelled; need at least 10"}
    both = [(a, b) for a, b in zip(r1, r2) if b is not None]
    human = [a if b is None else int(a and b) for a, b in zip(r1, r2)]
    in_plan = [h for h, m in zip(human, model) if m]
    held = [h for h, m in zip(human, model) if not m]
    return {"available": True, "labelled": len(r1), "second_rater": len(both),
            "precision_at_10": round(sum(top10) / len(top10), 3) if top10 else None, "top10_labelled": len(top10),
            "plan_precision": round(sum(in_plan) / len(in_plan), 3) if in_plan else None,
            "held_correctly_rejected": round(1 - sum(held) / len(held), 3) if held else None,
            "kappa_tool_vs_human": kappa(model, human),
            "kappa_rater1_vs_rater2": kappa([a for a, _ in both], [b for _, b in both]) if len(both) >= 5 else None}


# ------------------------------------------------------------------------------------------- driver
def evaluate_run(out_dir: Path, compare: Sequence[Path] = (), labels_dir: Optional[Path] = None) -> Dict[str, Any]:
    promos, meta, audit = _load(out_dir)
    labels_dir = labels_dir or (config.PROJECT_ROOT / "labels")
    weights = meta.get("weights") or config.DEFAULT_WEIGHTS
    judge = second_judge(promos, labels_dir)
    rep: Dict[str, Any] = {"scorer": meta.get("provider"), "promos": len(promos),
                           "second_judge": judge,
                           "repetition": repetition(promos, meta, labels_dir),
                           "scorer_agreement": scorer_agreement(promos, weights),
                           "comparisons": [],
                           "real_ads": real_ads(promos, judge),
                           "rule_checks": rule_checks(promos, meta, audit),
                           "human_labels": human_labels(promos, labels_dir)}
    J, _ = load_judge(labels_dir)
    for c in compare:
        other, ometa, _ = _load(Path(c))
        rep["comparisons"].append(compare_runs(promos, other, Path(c).name, J, ometa))
    (out_dir / "quality_eval.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "quality_eval.md").write_text(to_markdown(rep), encoding="utf-8")
    return rep


PLAN_NAMES = {"before_gemini:wave1": "Before fixes: Gemini run, Wave 1",
              "before_heuristic:wave1": "Before fixes: keyword run, Wave 1"}


def _jrow(name: str, m: Dict[str, Any]) -> str:
    if not m.get("judged"):
        return f"| {name} | {m.get('posts', 0)} | 0 | - | - | - | - |"
    return (f"| {name} | {m['posts']} | {m['judged']} | **{m['post_score']:.2f}** | {m['all_three_tests']}/{m['judged']} | "
            f"{m['weak_or_closed_endings']} | {m['weak_hooks']} |")


def to_markdown(rep: Dict[str, Any]) -> str:
    L = [f"# Quality evaluation - scored by {rep['scorer']}\n"]
    j = rep["second_judge"]
    L.append("## 1. Second judge (blind)\n")
    if j.get("available"):
        L.append(j["provenance"] + "\n")
        L.append("Post score: would post = 1, maybe = 0.5, no = 0, averaged. All three tests = strong hook, open ending "
                 "and makes sense alone.\n")
        L.append("| List | Posts | Judged | Post score | All three tests | Soft/closed endings | Weak hooks |")
        L.append("|---|---|---|---|---|---|---|")
        for k, m in j["this_run"].items():
            L.append(_jrow(f"This run: {k}", m))
        L.append(_jrow("This run: plan posts labelled after the main fixes (rounds 4-5)",
                       j["this_run_labelled_after_main_fixes"]))
        for k, m in j["earlier_plans"].items():
            L.append(_jrow(PLAN_NAMES.get(k, k), m))
        h = j["held_for_failing_a_test"]
        L.append(f"\nPromos held for failing a test, judged: {h['judged']}. The judge would not post "
                 f"{h['judge_would_not_post']}, says maybe to {h['judge_maybe']} and would post {h['judge_would_post']}.\n")
        if j["plan_promos_the_judge_faults"]:
            L.append("Plan posts the judge faults (where the tool picks badly):\n")
            for b in j["plan_promos_the_judge_faults"]:
                L.append(f"- {b['list']} #{b['post_order']} {b['promo_id']} ({b['clip']}): {b['judge']}")
        if j["held_promos_the_judge_would_post"]:
            L.append("\nHeld promos the judge would post (rules too strict):\n")
            for b in j["held_promos_the_judge_would_post"]:
                L.append(f"- {b['promo_id']} ({b['clip']}): held because {b['held_because']}")
        if j["unjudged_plan_promos"]:
            L.append(f"\nNot judged yet: {', '.join(j['unjudged_plan_promos'])}")
        L.append("")
    else:
        L.append(j.get("note", "") + "\n")
    r = rep["repetition"]
    L.append("## 2. Repetition\n")
    if r.get("available"):
        L.append("| List | Posts | Episodes | Story sequences | Most posts from one sequence |")
        L.append("|---|---|---|---|---|")
        rows = list(r["this_run"].items()) + [(PLAN_NAMES.get(k, k), m) for k, m in r["earlier_plans"].items()]
        for k, m in rows:
            eps = r["sequence_episodes"].get(m["that_sequence"], [])
            span = (f", Ep {eps[0]}" + (f"-{eps[-1]}" if eps[-1] != eps[0] else "")) if eps else ""
            L.append(f"| {k} | {m['posts']} | {m['episodes']} | {m['story_sequences']} | {m['most_from_one_sequence']} "
                     f"({m['that_sequence']}{span}) |")
        L.append("")
    L.append("## 3. Other runs on the same candidates\n")
    a = rep["scorer_agreement"]
    if a.get("available"):
        L.append(f"Model re-rank vs code score on the shortlist: rank correlation **{a['composite_spearman']}** over "
                 f"{a['promos']} promos; "
                 f"top-10 overlap **{a['top10_overlap']}/10**.\n")
    if rep["comparisons"]:
        L.append("| Compared with | Scorer | Shared clips | Score Spearman | Same plan decision | Post first overlap | "
                 "Top 15 overlap | Judge: their Post first | Judge: their top 15 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in rep["comparisons"]:
            pf, t15 = c.get("judge_other_post_first", {}), c.get("judge_other_top15", {})
            L.append(f"| {c['label']}{' (cache-only preview)' if c['cache_only_preview'] else ''} | {c['scorer']} | "
                     f"{c['shared_clips']} | {c.get('score_spearman')} | {c.get('same_plan_decision_pct')}% | "
                     f"{c['post_first_overlap']}/5 | {c['top15_overlap']}/15 | {pf.get('post_score', '-')} "
                     f"({pf.get('judged', 0)} judged) | {t15.get('post_score', '-')} ({t15.get('judged', 0)} judged) |")
        L.append("")
    elif not a.get("available"):
        L.append(a.get("note", "") + ". Compare: `python scripts/quality_eval.py --compare <other output dir>`\n")
    p = rep["real_ads"]
    L.append("## 4. The plan next to real short-drama ads\n")
    if p.get("plan"):
        L.append("| Pattern | Real ads | This plan | Source |\n|---|---|---|---|")
        for row in p["rows"]:
            L.append(f"| {row['pattern']} | {row['real_ads']} | {row['this_plan']} | {row['source']} |")
        L.append("\n" + p["caveat"] + "\n")
        for k, s in p["sources"].items():
            L.append(f"- {k}: [{s['title']}]({s['url']})")
        L.append("")
    L.append("## 5. Rule checks (show the rules ran, not that the picks are good)\n")
    for g in rep["rule_checks"]:
        L.append(f"- **{g['rule']}**: {g['effect']}")
    h = rep["human_labels"]
    L.append("\n## 6. Human labels\n")
    if h.get("available"):
        L.append(f"{h['labelled']} promos labelled ({h['second_rater']} by two people). Precision@10 **{h['precision_at_10']}**; "
                 f"plan precision **{h['plan_precision']}**; held promos correctly rejected **{h['held_correctly_rejected']}**; "
                 f"kappa tool-vs-human **{h['kappa_tool_vs_human']}**, rater-vs-rater **{h['kappa_rater1_vs_rater2']}**.")
    else:
        L.append(h.get("note", "") + " (labels/README.md)")
    return "\n".join(L) + "\n"
