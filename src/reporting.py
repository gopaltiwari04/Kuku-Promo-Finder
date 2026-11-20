"""Output writers + hard verification (shared by run_pipeline.py and the Streamlit app)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import config
from .parser import EpisodeIndex, ts_to_sec
from .schema import Episode, Promo, sec_to_ts

CSV_COLUMNS = ["PostOrder", "Wave", "WaveLabel", "PostReason", "PromoID", "Type", "Episodes", "Segments",
               "Seg1_Episode", "Seg1_Start", "Seg1_End", "Seg2_Episode", "Seg2_Start", "Seg2_End",
               "TotalDurationSec", "Score", "Hook", "Cliffhanger", "WhyItWorks", "HookStrength", "CliffhangerTension",
               "MakesSenseAlone", "PacingDensity", "TropeVirality", "SpoilerRisk", "NeedsHumanReview", "ReviewReasons",
               "WaveReason", "VariantOf", "ScoreRank", "Audience", "SellsPremise", "CodeScore", "Stage1Rank",
               "RerankedBy"]


def segments_str(p: Promo) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start_timestamp']}-{s['end_timestamp']} ({s['duration_sec']:.2f}s)"
                      for s in p.segments)


def promo_row(p: Promo) -> Dict[str, Any]:
    s1 = p.segments[0]
    s2 = p.segments[1] if len(p.segments) > 1 else {}
    sc = p.scores
    sig = p.signals or {}
    return {
        "PostOrder": p.post_order, "Wave": p.wave, "WaveLabel": p.wave_label, "PostReason": p.post_reason,
        "PromoID": p.promo_id, "Type": p.promo_type, "Episodes": "+".join(map(str, p.episodes)),
        "Segments": segments_str(p), "Seg1_Episode": s1["episode"], "Seg1_Start": s1["start_timestamp"],
        "Seg1_End": s1["end_timestamp"], "Seg2_Episode": s2.get("episode", ""), "Seg2_Start": s2.get("start_timestamp", ""),
        "Seg2_End": s2.get("end_timestamp", ""), "TotalDurationSec": f"{p.total_duration_sec:.2f}",
        "Score": f"{p.composite_score:.2f}", "Hook": p.hook, "Cliffhanger": p.cliffhanger, "WhyItWorks": p.why_it_works,
        "HookStrength": sc["hook_strength"], "CliffhangerTension": sc["cliffhanger_tension"],
        "MakesSenseAlone": sc["standalone_clarity"], "PacingDensity": sc["pacing_density"],
        "TropeVirality": sc["trope_virality"], "SpoilerRisk": sc["spoiler_risk"],
        "NeedsHumanReview": "yes" if p.needs_human_review else "no", "ReviewReasons": " | ".join(p.review_reasons),
        "WaveReason": p.wave_reason, "VariantOf": p.variant_of, "ScoreRank": p.score_rank,
        "Audience": sig.get("audience", ""),
        "SellsPremise": ("yes (montage)" if sig.get("premise_montage") else "yes") if sig.get("premise_reveal") else "",
        "CodeScore": sig.get("code_score", ""), "Stage1Rank": sig.get("stage1_rank", ""),
        "RerankedBy": sig.get("reranked_by", "") if sig.get("shortlisted") else "not shortlisted",
    }


def write_csv(path: Path, promos: Sequence[Promo]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for p in sorted(promos, key=lambda x: x.post_order):
            w.writerow(promo_row(p))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def episode_stats(episodes: Sequence[Episode], index: Dict[int, EpisodeIndex], promos: Sequence[Promo],
                  windows: Dict[int, int]) -> List[Dict[str, Any]]:
    rows = []
    for ep in episodes:
        ix = index[ep.episode_number]
        dl = ep.dialogue
        cover = 0.0
        cur_s = cur_e = None
        for d in sorted(dl, key=lambda x: x.start_sec):
            if cur_e is None or d.start_sec > cur_e:
                if cur_e is not None:
                    cover += cur_e - cur_s
                cur_s, cur_e = d.start_sec, d.end_sec
            else:
                cur_e = max(cur_e, d.end_sec)
        if cur_e is not None:
            cover += cur_e - cur_s
        mine = [p for p in promos if ep.episode_number in p.episodes]
        top = max(mine, key=lambda p: p.composite_score, default=None)
        rows.append({
            "episode": ep.episode_number, "filename": ep.filename, "duration_sec": ep.duration_sec,
            "duration_ts": sec_to_ts(ep.duration_sec), "scenes": len(ep.scenes),
            "blocks_total": len(ep.events),
            "camera_blocks": sum(e.track == "CAMERA" for e in ep.events),
            "action_blocks": sum(e.track == "ACTION" for e in ep.events),
            "dialogue_blocks": len(dl), "sound_fx_blocks": sum(e.track == "SOUND_FX" for e in ep.events),
            "speakers": len({d.speaker for d in dl}), "dialogue_words": sum(len(d.text.split()) for d in dl),
            "dialogue_coverage_pct": round(100 * cover / max(ep.duration_sec, 1e-6), 1),
            "lines_per_minute": round(len(dl) / max(ep.duration_sec, 1e-6) * 60, 1),
            "first_line_sec": round(dl[0].start_sec, 2) if dl else None,
            "clean_cut_starts": len(ix.clean_starts), "clean_cut_ends": len(ix.clean_ends),
            "windows_evaluated": windows.get(ep.episode_number, 0),
            "promos_including_episode": len(mine),
            "top_promo_id": top.promo_id if top else "", "top_promo_score": top.composite_score if top else "",
            "top_promo_wave": top.wave_label if top else "", "parse_warnings": len(ep.parse_warnings),
        })
    return rows


def write_report_md(path: Path, promos: Sequence[Promo], meta: Dict[str, Any], audit: Dict[str, Any]) -> None:
    ranked = sorted(promos, key=lambda x: x.post_order)
    waves = {w: [p for p in ranked if p.wave == w] for w in (1, 2, 3, 4)}
    rk = meta.get("ranking") or {}
    R = config.POSTING_RULES
    L: List[str] = []
    L.append(f"# Promo posting plan - *{meta['show_title']}*\n")
    L.append(f"Scored by **{meta['provider']}** · {meta['episodes_parsed']} episodes ({meta['corpus_runtime_min']:.1f} min) "
             f"· {len(promos)} promos found · {sum(len(waves[w]) for w in (1, 2, 3))} in the posting plan · "
             f"{len(waves[4])} held. Every timestamp is copied from the screenplay; length = end - start.\n")
    L.append("## Which promos go out first, and why\n")
    L.append("Every posted promo passes the brief's three tests (a hook in the first 3 s, an ending left open, "
             f"makes sense alone >= {R['min_clarity']}/10). Inside each list the order is the score; no two posts "
             f"share footage or continue each other, and no list holds more than {R['max_per_sequence']} posts from one "
             "story sequence.\n")
    L.append("| List | Promos | Rule |")
    L.append("|---|---|---|")
    L.append(f"| Post first | {len(waves[1])} | best {R['post_first']} from the first quarter of the show "
             f"(Ep 1-{rk.get('first_quarter_last_episode', '?')}), spoiler <= {R['first_max_spoiler']}; one slot may go "
             "to a title reveal from any episode |")
    L.append(f"| Post next | {len(waves[2])} | next {R['post_next']} by score, spoiler <= {R['next_max_spoiler']} |")
    L.append(f"| Later | {len(waves[3])} | up to {R['later']} more; spoiler above {R['next_max_spoiler']} needs a check |")
    L.append(f"| Hold | {len(waves[4])} | failed a test, an alternate of a posted promo, or below the top 40 "
             "(reason on each) |\n")

    def block(p: Promo) -> None:
        L.append(f"### #{p.post_order} `{p.promo_id}` - score {p.composite_score:.1f} - {p.total_duration_sec:.1f} s")
        L.append(f"Segments: {segments_str(p)}  ")
        L.append(f"Posted here because: {p.post_reason}\n")
        L.append(f"- **Hook (first 3 s):** {p.hook}")
        L.append(f"- **Cliffhanger:** {p.cliffhanger}")
        L.append(f"- **Why it works:** {p.why_it_works}")
        if p.review_reasons:
            L.append(f"- Check before posting: {'; '.join(p.review_reasons)}")
        L.append("")

    for w in (1, 2, 3):
        if waves[w]:
            L.append(f"## {config.WAVE_LABELS[w]}\n")
            for p in waves[w]:
                block(p)
    L.append("## Held promos (top 15 by score)\n")
    L.append("| Promo | Segments | Score | Why held |")
    L.append("|---|---|---|---|")
    for p in sorted(waves[4], key=lambda x: -x.composite_score)[:15]:
        L.append(f"| `{p.promo_id}` | {segments_str(p)} | {p.composite_score:.1f} | {p.wave_reason} |")
    L.append(f"\nAll {len(waves[4])} held promos with reasons: `held_promos.csv`.\n")
    L.append("## Where the tool picks badly (see `quality_audit_and_bad_picks.json`)\n")
    for c in audit["failure_modes"]:
        L.append(f"- **{c['failure_mode']}** ({c['category'].replace('_', ' ')}): {c['title']}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ANON_TOKEN_RX = re.compile(r"\b(?:speaker|SPEAKER|Speaker)[ _]?[A-Z]\b|\[someone\]")
COPY_FORMAT_RX = {"hook": re.compile(r"^\[(?:Ep\d+ )?\d{2}:\d{2}\.\d{2}\] "),
                  "cliffhanger": re.compile(r"^\[cut (?:Ep\d+ )?\d{2}:\d{2}\.\d{2}\] ")}


def verify(promos: Sequence[Promo], episodes: Sequence[Episode], errors: Sequence[Dict[str, str]], n_files: int,
           csv_path: Path, json_path: Path, hashes_before: Dict[str, str], min_d: float, max_d: float,
           generator: Any = None, provider_label: str = "heuristic:heuristic-v1",
           cache_only: bool = False) -> Dict[str, Any]:
    by_ep = {e.episode_number: e for e in episodes}
    raw = {e.episode_number: Path(e.path).read_text(encoding="utf-8-sig", errors="replace") for e in episodes}
    checks: Dict[str, Any] = {}
    checks["episodes_parsed"] = {"pass": len(episodes) == n_files and not errors,
                                 "detail": f"{len(episodes)}/{n_files} parsed, {len(errors)} errors"}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    jpromos = data["promos"]
    bad_dur = [p["promo_id"] for p in jpromos if not (min_d - 1e-6 <= p["total_duration_sec"] <= max_d + 1e-6)]
    rows = []
    for cp in (csv_path, csv_path.with_name("held_promos.csv")):
        if cp.exists():
            with open(cp, encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
    bad_csv = [r["PromoID"] for r in rows if not (min_d - 1e-6 <= float(r["TotalDurationSec"]) <= max_d + 1e-6)]
    checks["durations_in_bounds_json"] = {"pass": not bad_dur, "detail": f"{len(jpromos) - len(bad_dur)}/{len(jpromos)} in [{min_d}, {max_d}]s", "violations": bad_dur}
    checks["durations_in_bounds_csv"] = {"pass": not bad_csv and len(rows) == len(jpromos),
                                         "detail": f"{len(rows) - len(bad_csv)}/{len(rows)} rows in bounds", "violations": bad_csv}
    missing, wallclock_bad, order_bad = [], [], []
    for p in jpromos:
        total = 0.0
        prev = None
        for s in p["segments"]:
            txt = raw.get(s["episode"], "")
            verbatim = lambda ts: f"({ts})" in txt or f": {ts}" in txt
            for k in ("start_timestamp", "end_timestamp"):
                ts = s[k]
                if verbatim(ts):
                    continue
                # a tightened start sits pre_roll seconds before a verbatim line: check the line and the gap
                anchor, pre = s.get("anchor_timestamp"), s.get("pre_roll_sec", 0.0)
                gap = ts_to_sec(anchor) - ts_to_sec(ts) if (k == "start_timestamp" and anchor) else None
                if not (gap is not None and verbatim(anchor) and abs(gap - pre) <= 0.011
                        and 0 < gap <= config.PRE_ROLL_SEC + 0.011):
                    missing.append(f"{p['promo_id']} Ep{s['episode']} {k}={ts}")
            d = round(ts_to_sec(s["end_timestamp"]) - ts_to_sec(s["start_timestamp"]), 2)
            if abs(d - s["duration_sec"]) > 0.011 or d <= 0:
                wallclock_bad.append(p["promo_id"])
            total += d
            if prev is not None and (s["episode"], ts_to_sec(s["start_timestamp"])) <= prev:
                order_bad.append(p["promo_id"])
            prev = (s["episode"], ts_to_sec(s["end_timestamp"]))
        if abs(round(total, 2) - p["total_duration_sec"]) > 0.011:
            wallclock_bad.append(p["promo_id"])
    n_ts = sum(2 * len(p["segments"]) for p in jpromos)
    n_pre = sum(bool(s.get("pre_roll_sec")) for p in jpromos for s in p["segments"])
    checks["timestamps_verbatim_in_source"] = {
        "pass": not missing, "violations": missing[:20],
        "detail": f"{n_ts - len(missing)}/{n_ts} segment timestamps found verbatim ({n_pre} starts sit "
                  f"<= {config.PRE_ROLL_SEC}s before a verbatim line)"}
    checks["wall_clock_duration_math"] = {"pass": not wallclock_bad, "detail": "segment & total durations == end - start", "violations": wallclock_bad}
    checks["segments_in_chronological_order"] = {"pass": not order_bad, "detail": "seg1 precedes seg2 (same or earlier episode)", "violations": order_bad}
    types = {p["promo_type"] for p in jpromos}
    want = {"continuous", "multi_segment_intra", "multi_segment_cross"} if len(episodes) > 1 else {"continuous"}
    checks["all_promo_types_present"] = {"pass": want <= types, "detail": ", ".join(f"{t}={sum(p['promo_type'] == t for p in jpromos)}" for t in sorted(types))}
    ranked = all(isinstance(p.get("composite_score"), (int, float)) and p.get("post_order", 0) > 0 for p in jpromos)
    checks["all_promos_scored_and_ranked"] = {"pass": ranked, "detail": f"{len(jpromos)} promos with composite + post_order"}
    req = ["promo_id", "promo_type", "segments", "total_duration_sec", "hook", "cliffhanger", "why_it_works", "scores"]
    miss_fields = [p.get("promo_id") for p in jpromos if any(not p.get(k) for k in req)]
    checks["required_fields_present"] = {"pass": not miss_fields, "detail": ", ".join(req), "violations": miss_fields}
    changed = [f for f, h in hashes_before.items() if sha256(Path(f)) != h]
    checks["episode_files_untouched"] = {"pass": not changed, "detail": f"sha256 identical for {len(hashes_before) - len(changed)}/{len(hashes_before)} source files", "violations": changed}
    eps_covered = sorted({e for p in jpromos for e in p["episodes"]})
    checks["every_episode_has_a_promo"] = {"pass": len(eps_covered) == len(episodes), "detail": f"{len(eps_covered)}/{len(episodes)} episodes"}
    # entity consistency: villain guard ran, and no shipped cross arc humbles a villain absent from its setup
    if generator is not None:
        guard_ran = hasattr(generator, "villain_swaps")
        viol = []
        for p in jpromos:
            if p["promo_type"] != "multi_segment_cross":
                continue
            a, b = p["segments"]
            sa, ea = ts_to_sec(a["start_timestamp"]), ts_to_sec(a["end_timestamp"])
            sb, eb = ts_to_sec(b["start_timestamp"]), ts_to_sec(b["end_timestamp"])
            new = generator._humbled_antagonists(b["episode"], sb, eb) - generator._entities(a["episode"], sa, ea)
            if new:
                viol.append(f"{p['promo_id']}: {sorted(new)}")
        n_cross = sum(p["promo_type"] == "multi_segment_cross" for p in jpromos)
        checks["entity_consistency_guard"] = {
            "pass": guard_ran and not viol,
            "detail": f"villain guard active ({len(getattr(generator, 'villain_swaps', []))} arcs blocked); "
                      f"{n_cross - len(viol)}/{n_cross} shipped cross arcs keep the same villain", "violations": viol}
    leaks = [p["promo_id"] for p in jpromos
             if ANON_TOKEN_RX.search(" ".join(str(p.get(k, "")) for k in ("hook", "cliffhanger", "why_it_works")))]
    checks["no_anonymised_names_in_copy"] = {
        "pass": not leaks, "detail": f"{len(jpromos) - len(leaks)}/{len(jpromos)} promos name characters instead of "
                                     "'Speaker A'", "violations": leaks}
    fmt_bad = [p["promo_id"] for p in jpromos if not all(rx.match(str(p.get(k, ""))) for k, rx in COPY_FORMAT_RX.items())]
    checks["copy_format_standard"] = {"pass": not fmt_bad, "detail": "hook = [timestamp] \"line\" - why; cliffhanger = "
                                      "[cut timestamp] \"line\" - why", "violations": fmt_bad}
    from .evaluator import CLIFF_Q, STOCK_RX
    from .candidate_generator import finale_first_episode
    from .ranker import clash, gate
    R = config.POSTING_RULES
    finale_from = finale_first_episode(sorted(by_ep), config.FINALE_SHARE)
    plan = sorted([p for p in promos if p.wave in (1, 2, 3)], key=lambda p: p.post_order)
    lists = {w: [p for p in plan if p.wave == w] for w in (1, 2, 3)}
    failing = [f"{p.promo_id}: {gate(p, R)}" for p in plan if gate(p, R)]
    checks["plan_passes_the_three_tests"] = {
        "pass": not failing, "detail": f"{len(plan)} posted promos: hook in the first 3 s, ending left open, makes "
                                       f"sense alone >= {R['min_clarity']}", "violations": failing}
    clashes = [f"{b.promo_id}: {clash(a, b, R['chain_gap_sec'])}" for i, a in enumerate(plan) for b in plan[i + 1:]
               if clash(a, b, R["chain_gap_sec"])]
    checks["no_repeats_in_plan"] = {"pass": not clashes, "detail": "no two posts share footage or start where another "
                                    "ends", "violations": clashes}
    over = []
    for w, grp in lists.items():
        cnt: Dict[str, int] = {}
        for p in grp:
            for sq in (p.signals or {}).get("sequences", []):
                cnt[sq] = cnt.get(sq, 0) + 1
        over += [f"{config.WAVE_LABELS[w]}: {sq} x{n}" for sq, n in cnt.items() if n > R["max_per_sequence"]]
    checks["at_most_two_per_story_sequence"] = {
        "pass": not over, "detail": f"no list holds more than {R['max_per_sequence']} posts from one story sequence",
        "violations": over}
    eps_all = sorted(by_ep)
    import math
    first_ep = (eps_all[0] - 1 + max(1, math.ceil(R["first_quarter"] * (eps_all[-1] - eps_all[0] + 1)))) if eps_all else 1
    bad_first = [p.promo_id for p in lists[1] if not (
        (max(p.episodes) <= first_ep and p.scores["spoiler_risk"] <= R["first_max_spoiler"])
        or (p.signals or {}).get("premise_reveal"))]
    n_premise = sum(1 for p in lists[1] if max(p.episodes) > first_ep)
    checks["post_first_rules"] = {
        "pass": len(lists[1]) <= R["post_first"] and not bad_first and n_premise <= R["premise_slots"],
        "detail": f"{len(lists[1])} posts (max {R['post_first']}), from Ep 1-{first_ep} with spoiler <= "
                  f"{R['first_max_spoiler']}, plus {n_premise} title reveal(s) (max {R['premise_slots']})",
        "violations": bad_first}
    unordered = [f"{config.WAVE_LABELS[w]}: {a.promo_id} {a.composite_score:.1f} < {b.promo_id} {b.composite_score:.1f}"
                 for w, grp in lists.items() for a, b in zip(grp, grp[1:]) if a.composite_score < b.composite_score - 1e-6]
    checks["order_follows_score"] = {"pass": not unordered, "detail": "inside each list, posting order = score order",
                                     "violations": unordered}
    allowed_q = [q.lower() for q in CLIFF_Q.values() if q]
    copy_bad = []
    for p in plan:
        txt = " ".join((p.hook, p.cliffhanger, p.why_it_works))
        if STOCK_RX.search(txt):
            copy_bad.append(f"{p.promo_id}: stock phrase \u201c{STOCK_RX.search(txt).group(0)}\u201d")
        spoken = " ".join(e.text for sg in p.segments for e in by_ep[sg["episode"]].events
                          if e.track == "DIALOGUE" and ts_to_sec(sg["start_timestamp"]) - 1e-6 <= e.start_sec
                          <= ts_to_sec(sg["end_timestamp"]) + 1e-6).lower()
        for q in re.findall(r"[^.?!\"\u201c\u201d\u2018\u2019'\u2014:]*\?", p.hook + " " + p.cliffhanger):
            q_norm = q.strip(" '\u2019\u2018:;,\u2014-").lower()
            if q_norm and q_norm not in spoken and not any(q_norm in a or a in q_norm for a in allowed_q):
                copy_bad.append(f"{p.promo_id}: a question the clip never asks (\u201c{q.strip()[:60]}\u201d)")
                break
    checks["copy_is_specific"] = {"pass": not copy_bad, "detail": "no stock marketing phrases and no invented "
                                  "questions in the plan's copy", "violations": copy_bad}
    # the premise and the finale
    premise_ok = [p for p in promos if (p.signals or {}).get("premise_reveal") and gate(p, R) is None
                  and max(p.episodes) < finale_from]
    in_first = [p.promo_id for p in lists[1] if (p.signals or {}).get("premise_reveal")]
    checks["post_first_sells_premise"] = {
        "pass": bool(in_first) or not premise_ok,
        "detail": (f"Post first carries {', '.join(in_first)}" if in_first else
                   "no promo in this show sells the title's twist and passes the tests" if not premise_ok else
                   f"{len(premise_ok)} premise promos pass the tests but none is in Post first")}
    cold_finale = [f"{config.WAVE_LABELS[w]}: {p.promo_id} (Ep {max(p.episodes)})" for w in (1, 2)
                   for p in lists[w] if max(p.episodes) >= finale_from]
    checks["no_finale_in_cold_lists"] = {
        "pass": not cold_finale, "violations": cold_finale,
        "detail": (f"finale = Ep {finale_from}-{eps_all[-1]} (last {config.FINALE_SHARE:.0%}); Post first and Post next "
                   "use none of it" if finale_from <= (eps_all[-1] if eps_all else 0) else
                   "show too short for a finale rule")}
    if not provider_label.startswith("heuristic"):
        short = [p for p in promos if (p.signals or {}).get("shortlisted")]
        kept = [p.promo_id for p in short if (p.signals or {}).get("reranked_by") != provider_label]
        checks["stage2_model_rerank_complete"] = {
            "pass": not kept, "violations": kept[:20],
            "detail": f"{len(short) - len(kept)}/{len(short)} shortlisted promos re-ranked by {provider_label}"
                      + (f"; {len(kept)} kept the keyword order (time budget, error or no cached answer)" if kept else "")}
    if cache_only:
        checks["real_model_run"] = {"pass": False, "detail": "cache-only preview: new promos were skipped; re-run with "
                                    "network access before using these outputs"}
    return {"all_pass": all(c["pass"] for c in checks.values()), "checks": checks}
