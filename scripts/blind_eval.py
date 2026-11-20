#!/usr/bin/env python3
"""Blind A/B evaluation: the pipeline's posting plan against a simple baseline, scored by judges who cannot see
which is which.

  Baseline   the last 40 s of an episode (what an editor grabs without reading the script: episodes end on a
             cliffhanger). One per episode, 15 episodes sampled with a fixed seed, finale episodes excluded.
  Pipeline   the 15 promos of Post first + Post next from the latest run (raw_outputs/).

Steps
  1. python scripts/blind_eval.py packet           30 clips, shuffled, ids C01-C30, source hidden
                                                    -> eval/packet.md + packet.json (screenplay text: git-ignored),
                                                       eval/key.json (which clip is which), eval/manual_labels.csv,
                                                       eval/JUDGE_PROMPT.md
  2. python scripts/blind_eval.py judge --provider gemini --model gemini-2.5-flash
                                                    Judge 1, a model through src/llm_adapter (cached in .llm_cache/)
                                                    -> eval/judge1_<model>.json
     Judge 2: paste eval/JUDGE_PROMPT.md + eval/packet.md into a different model; save its JSON as
              eval/judge2_<model>.json (the model name in the file name is the name shown in the doc)
     Person:  fill eval/manual_labels.csv while reading eval/packet.md
  3. python scripts/blind_eval.py merge            -> eval/results.json + eval/results.md
     Per judge and source: mean hook / ending / makes-sense-alone (1-5), would-post rate (Y=1, M=0.5, N=0),
     the pipeline-minus-baseline difference, the pipeline's win rate over all pipeline x baseline pairs, and how
     often the judges agree.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.candidate_generator import finale_first_episode  # noqa: E402
from src.parser import load_corpus, ts_to_sec  # noqa: E402
from src.schema import Episode  # noqa: E402

EVAL = ROOT / "eval"
BASELINE_SEC = 40.0
N_EACH = 15
DIMS = ("hook", "ending", "clarity")
POST_VALUE = {"Y": 1.0, "M": 0.5, "N": 0.0}

JUDGE_PROMPT = """You are judging short clips cut from a vertical microdrama (the kind ReelShort and DramaBox advertise \
on TikTok, Reels and Shorts). Each clip is written out as screenplay blocks with the time since the clip starts: \
CAMERA and ACTION are what you see, SOUND is what you hear, and NAME: "line" is what is said. "--- jump cut ---" \
marks a cut to a later moment.

Imagine each clip is posted as a paid ad to people who have never heard of the show. Score every clip on its own:

  hook      1-5  would a stranger scrolling past stop in the first 3 seconds? (5 = a slap, a shocking line, a reveal;
                 1 = nothing happens yet)
  ending    1-5  does it cut on an unanswered moment that makes you need the next part? (5 = you must know what
                 happens; 1 = the scene is settled or it just stops)
  clarity   1-5  does it make sense to someone who has never seen the show? (5 = who wants what and what is at
                 risk is clear; 1 = lost)
  would_post  Y / M / N   would you spend ad money on it as it is?
  note      one short reason

Be strict and use the whole scale. Do not guess how the clips were chosen.

Return ONLY JSON: {"clips": [{"id": "C01", "hook": 3, "ending": 4, "clarity": 2, "would_post": "M", "note": "..."}]}"""


# ------------------------------------------------------------------------------------------- packet
def _render(ep: Episode, start: float, end: float, t0: float) -> List[str]:
    out = []
    for e in sorted(ep.events, key=lambda x: (x.start_sec, x.idx)):
        if e.start_sec < start - 1e-6 or e.start_sec >= end - 1e-6:
            continue
        t = f"[+{e.start_sec - t0:04.1f}s]"
        if e.track == "DIALOGUE":
            tag = f" ({', '.join(e.emotion_tags)})" if e.emotion_tags else ""
            out.append(f'{t} {e.speaker_raw.title()}{tag}: "{e.text}"')
        else:
            out.append(f"{t} {e.track.replace('_', ' ')}: {e.text}")
    return out


def clip_text(by_ep: Dict[int, Episode], segments: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
    lines: List[str] = []
    t_off = 0.0
    for k, s in enumerate(segments):
        a, b = ts_to_sec(s["start_timestamp"]), ts_to_sec(s["end_timestamp"])
        if k:
            lines.append("--- jump cut ---")
        lines += _render(by_ep[s["episode"]], a, b, a - t_off)
        t_off += b - a
    return "\n".join(lines), round(t_off, 2)


def baseline_segment(ep: Episode, seconds: float = BASELINE_SEC) -> Dict[str, Any]:
    """The last `seconds` of the episode, starting on the first block that begins inside that window (so both
    timestamps are verbatim in the screenplay)."""
    end_ev = max(ep.events, key=lambda e: e.end_sec)
    end = end_ev.end_sec
    first = min((e for e in ep.events if e.start_sec >= end - seconds - 1e-6), key=lambda e: e.start_sec)
    return {"episode": ep.episode_number, "start_timestamp": first.start_ts, "end_timestamp": end_ev.end_ts,
            "duration_sec": round(end - first.start_sec, 2)}


def _answers_exist() -> List[str]:
    """Judge files or filled-in manual labels that a new packet (new ids) would silently invalidate."""
    found = [p.name for p in EVAL.glob("judge*.json")]
    manual = EVAL / "manual_labels.csv"
    if manual.exists():
        with open(manual, encoding="utf-8") as fh:
            if any(r.get("hook", "").strip() for r in csv.DictReader(fh)):
                found.append(manual.name)
    return found


def make_packet(run_dir: Path, input_dir: Path, seed: int, force: bool = False) -> Dict[str, Any]:
    if (EVAL / "key.json").exists() and _answers_exist() and not force:
        raise SystemExit(f"eval/ already holds answers for the current packet ({', '.join(_answers_exist())}); a new "
                         "packet would reshuffle the ids. Move them aside or pass --force.")
    episodes, _ = load_corpus(input_dir)
    by_ep = {e.episode_number: e for e in episodes}
    finale = finale_first_episode(sorted(by_ep), config.FINALE_SHARE)
    promos = json.loads((run_dir / "promos_all_episodes.json").read_text(encoding="utf-8"))["promos"]
    plan = sorted([p for p in promos if p["wave"] in (1, 2)], key=lambda p: p["post_order"])
    if len(plan) < N_EACH:                                   # top up from Later (cold-safe only)
        plan += sorted([p for p in promos if p["wave"] == 3 and max(p["episodes"]) < finale],
                       key=lambda p: p["post_order"])[:N_EACH - len(plan)]
    plan = plan[:N_EACH]
    rng = random.Random(seed)
    base_eps = sorted(rng.sample([e for e in sorted(by_ep) if e < finale], min(N_EACH, len(by_ep))))
    items = [{"source": "pipeline", "promo_id": p["promo_id"], "post_order": p["post_order"],
              "segments": [{k: s[k] for k in ("episode", "start_timestamp", "end_timestamp", "duration_sec")}
                           for s in p["segments"]]} for p in plan]
    items += [{"source": "baseline", "promo_id": f"LAST40_EP{e:02d}", "segments": [baseline_segment(by_ep[e])]}
              for e in base_eps]
    rng.shuffle(items)
    clips, key = [], {}
    for i, it in enumerate(items, 1):
        cid = f"C{i:02d}"
        text, dur = clip_text(by_ep, it["segments"])
        clips.append({"id": cid, "duration_sec": dur, "text": text})
        key[cid] = dict(it, duration_sec=dur)
    EVAL.mkdir(exist_ok=True)
    (EVAL / "packet.json").write_text(json.dumps({"clips": clips}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Blind clip packet\n", f"{len(clips)} clips from one show, in random order. Score each one on its own "
          "(see JUDGE_PROMPT.md).\n"]
    for c in clips:
        md += [f"## {c['id']} ({c['duration_sec']:.0f} s)\n", "```", c["text"], "```\n"]
    (EVAL / "packet.md").write_text("\n".join(md), encoding="utf-8")
    (EVAL / "key.json").write_text(json.dumps({"seed": seed, "run": str(run_dir.name), "baseline_sec": BASELINE_SEC,
                                               "finale_first_episode": finale, "clips": key}, indent=1),
                                   encoding="utf-8")
    (EVAL / "JUDGE_PROMPT.md").write_text(JUDGE_PROMPT + "\n", encoding="utf-8")
    manual = EVAL / "manual_labels.csv"
    if force or not _answers_exist():
        with open(manual, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "hook", "ending", "clarity", "would_post", "note"])
            for c in clips:
                w.writerow([c["id"], "", "", "", "", ""])
    return {"clips": len(clips), "pipeline": sum(v["source"] == "pipeline" for v in key.values()),
            "baseline": sum(v["source"] == "baseline" for v in key.values()), "baseline_episodes": base_eps}


# ------------------------------------------------------------------------------------------- judge
def _norm_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        row = {d: max(1, min(5, int(round(float(r[d]))))) for d in DIMS}
    except (KeyError, TypeError, ValueError):
        return None
    wp = str(r.get("would_post", "")).strip().upper()[:1]
    row["would_post"] = wp if wp in POST_VALUE else "N"
    row["note"] = str(r.get("note", ""))[:200]
    return row


def run_judge(provider: str, model: Optional[str], batch: int, use_cache: bool) -> Path:
    from src.llm_adapter import LLMAdapter
    clips = json.loads((EVAL / "packet.json").read_text(encoding="utf-8"))["clips"]
    ad = LLMAdapter(provider, model, log=print)
    scores: Dict[str, Any] = {}
    for g in range(0, len(clips), batch):
        group = clips[g:g + batch]
        user = "\n\n".join(f"### {c['id']} ({c['duration_sec']:.0f} s)\n{c['text']}" for c in group)
        data, hit = ad.complete_json_cached(JUDGE_PROMPT, user + "\n\nReturn the JSON object only.", 4096,
                                            tag=f"judge {g // batch + 1}", use_cache=use_cache)
        for r in (data or {}).get("clips", []):
            row = _norm_row(r) if isinstance(r, dict) else None
            if row and str(r.get("id")) in {c["id"] for c in group}:
                scores[str(r["id"])] = row
        print(f"  clips {g + 1}-{g + len(group)}: {'cached' if hit else 'scored'}")
    out = EVAL / f"judge1_{re.sub(r'[^A-Za-z0-9._-]+', '_', ad.model or provider)}.json"
    out.write_text(json.dumps({"judge": ad.label, "clips": [dict(id=k, **v) for k, v in sorted(scores.items())]},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(scores)}/{len(clips)} clips scored -> {out.relative_to(ROOT)}")
    return out


# ------------------------------------------------------------------------------------------- merge
def load_json_judge(path: Path) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("clips", d) if isinstance(d, dict) else d
    out = {}
    for r in rows:
        row = _norm_row(r) if isinstance(r, dict) else None
        if row:
            out[str(r["id"]).strip().upper()] = row
    # judge name: the file's "judge" field, else the file name ("judge2_chatgpt-5.6-terra.json" -> "chatgpt-5.6-terra")
    name = (d.get("judge") if isinstance(d, dict) else None) or re.sub(r"^judge\d*_?", "", path.stem) or path.stem
    return name, out


def load_manual(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = _norm_row(r)
            if row:
                out[r["id"].strip().upper()] = row
    return out


def _mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 2) if xs else None


def _total(r: Dict[str, Any]) -> float:
    return sum(r[d] for d in DIMS) + 5 * POST_VALUE[r["would_post"]]


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    from src.quality_eval import spearman as sp
    return sp(a, b)


def judge_summary(rows: Dict[str, Dict[str, Any]], key: Dict[str, Any]) -> Dict[str, Any]:
    by_src = {s: [r for cid, r in rows.items() if key.get(cid, {}).get("source") == s] for s in ("pipeline", "baseline")}
    res: Dict[str, Any] = {"clips_scored": len(rows)}
    for s, rs in by_src.items():
        res[s] = {d: _mean([r[d] for r in rs]) for d in DIMS}
        res[s]["would_post"] = _mean([POST_VALUE[r["would_post"]] for r in rs])
        res[s]["n"] = len(rs)
    res["delta"] = {k: (round(res["pipeline"][k] - res["baseline"][k], 2)
                        if res["pipeline"][k] is not None and res["baseline"][k] is not None else None)
                    for k in DIMS + ("would_post",)}
    pairs = [(_total(p), _total(b)) for p in by_src["pipeline"] for b in by_src["baseline"]]
    res["win_rate"] = round(sum(1.0 if x > y else 0.5 if x == y else 0.0 for x, y in pairs) / len(pairs), 3) if pairs else None
    return res


def merge(judge1: Optional[Path], judge2: Path, manual: Path) -> Dict[str, Any]:
    key = json.loads((EVAL / "key.json").read_text(encoding="utf-8"))["clips"]
    judges: Dict[str, Dict[str, Dict[str, Any]]] = {}
    j1 = [judge1] if judge1 else sorted(EVAL.glob("judge1_*.json"))
    for i, p in enumerate(j1):
        name, rows = load_json_judge(p)
        judges[f"Judge 1 ({name})" if len(j1) == 1 else f"Judge 1.{i + 1} ({name})"] = rows
    j2 = [judge2] if judge2 else sorted(EVAL.glob("judge2*.json"))
    for i, p in enumerate(j2):
        if p.exists():
            name, rows = load_json_judge(p)
            judges[f"Judge 2 ({name})" if len(j2) == 1 else f"Judge 2.{i + 1} ({name})"] = rows
    m = load_manual(manual)
    if m:
        judges["Person (manual labels)"] = m
    rep: Dict[str, Any] = {"clips": len(key), "judges": {}, "agreement": [], "pooled": None}
    if not judges:
        return rep                                   # nothing to write yet
    for name, rows in judges.items():
        rep["judges"][name] = judge_summary(rows, key)
    for (na, ra), (nb, rb) in combinations(judges.items(), 2):
        shared = sorted(set(ra) & set(rb))
        if len(shared) >= 5:
            rep["agreement"].append({
                "pair": f"{na} vs {nb}", "clips": len(shared),
                "total_spearman": spearman([_total(ra[c]) for c in shared], [_total(rb[c]) for c in shared]),
                "same_would_post_pct": round(100 * sum(ra[c]["would_post"] == rb[c]["would_post"] for c in shared)
                                             / len(shared), 1)})
    if judges:
        pooled: Dict[str, Dict[str, Any]] = {}
        for cid in key:
            rs = [rows[cid] for rows in judges.values() if cid in rows]
            if rs:
                pooled[cid] = {d: sum(r[d] for r in rs) / len(rs) for d in DIMS}
                pooled[cid]["would_post"] = max(POST_VALUE, key=lambda k: -abs(POST_VALUE[k] - sum(
                    POST_VALUE[r["would_post"]] for r in rs) / len(rs)))
        rep["pooled"] = judge_summary(pooled, key)
        rep["per_clip"] = [{"id": cid, "source": key[cid]["source"], "promo_id": key[cid]["promo_id"],
                            "segments": key[cid]["segments"],
                            **{name: rows.get(cid) for name, rows in judges.items()}} for cid in sorted(key)]
    (EVAL / "results.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    (EVAL / "results.md").write_text(to_markdown(rep), encoding="utf-8")
    return rep


def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# Blind A/B: pipeline posting plan vs last 40 s of an episode\n",
         f"{rep['clips']} clips, shuffled, source hidden from every judge. Scores are 1-5; would-post is Y=1, M=0.5, "
         "N=0. Win rate = share of pipeline x baseline pairs where the pipeline clip scores higher (ties = half).\n",
         "| Judge | Source | n | Hook | Ending | Makes sense alone | Would post |", "|---|---|---|---|---|---|---|"]
    rows = list(rep["judges"].items()) + ([("All judges pooled", rep["pooled"])] if rep.get("pooled") else [])
    for name, s in rows:
        for src in ("pipeline", "baseline"):
            v = s[src]
            L.append(f"| {name} | {src} | {v['n']} | {v['hook']} | {v['ending']} | {v['clarity']} | {v['would_post']} |")
        d = s["delta"]
        L.append(f"| {name} | **difference** | | **{d['hook']:+}** | **{d['ending']:+}** | **{d['clarity']:+}** | "
                 f"**{d['would_post']:+}** (win rate {s['win_rate']}) |" if None not in d.values() else
                 f"| {name} | difference | | - | - | - | - |")
    if rep["agreement"]:
        L += ["\n## Do the judges agree?\n", "| Pair | Clips | Rank correlation (total) | Same would-post |",
              "|---|---|---|---|"]
        L += [f"| {a['pair']} | {a['clips']} | {a['total_spearman']} | {a['same_would_post_pct']}% |"
              for a in rep["agreement"]]
    L.append("\nThe clips and which is which: `eval/key.json`. Judge prompt: `eval/JUDGE_PROMPT.md`.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("packet")
    a.add_argument("--run", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    a.add_argument("--input_dir", type=Path, default=config.DEFAULT_INPUT_DIR)
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--force", action="store_true", help="rebuild even if judges' answers exist (ids change)")
    j = sub.add_parser("judge")
    j.add_argument("--provider", default="gemini", choices=["gemini", "openai", "anthropic"])
    j.add_argument("--model", default=None)
    j.add_argument("--batch", type=int, default=10)
    j.add_argument("--no_cache", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--judge1", type=Path, default=None, help="default: every eval/judge1_*.json")
    m.add_argument("--judge2", type=Path, default=None, help="default: every eval/judge2*.json")
    m.add_argument("--manual", type=Path, default=EVAL / "manual_labels.csv")
    args = ap.parse_args(argv)
    if args.cmd == "packet":
        info = make_packet(args.run, args.input_dir, args.seed, args.force)
        print(f"{info['clips']} clips ({info['pipeline']} pipeline, {info['baseline']} baseline from episodes "
              f"{info['baseline_episodes']}) -> eval/packet.md, eval/packet.json, eval/key.json, "
              "eval/manual_labels.csv, eval/JUDGE_PROMPT.md")
    elif args.cmd == "judge":
        run_judge(args.provider, args.model, args.batch, not args.no_cache)
    else:
        rep = merge(args.judge1, args.judge2, args.manual)
        if not rep["judges"]:
            print("no judge files yet: run `judge`, save eval/judge2_<model>.json, or fill eval/manual_labels.csv")
            return 1
        print((EVAL / "results.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
