#!/usr/bin/env python3
"""Kuku Problem 1 - find, stitch, rank and audit microdrama promos across every episode.

Two stages: code cuts, scores and shortlists ~100 promos; a re-ranker (keyword by default, any model on request)
orders the shortlist and writes the why-it-works line. Code then builds the posting plan.

Zero-config:   python run_pipeline.py
With a model:  python run_pipeline.py --input_dir "<folder>" --output_dir ./raw_outputs \
                   --provider gemini --model gemini-2.5-flash --min_duration 20 --max_duration 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config
from src.llm_adapter import LLMError
from src.pipeline import run_folder
from src.reporting import segments_str


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=config.DEFAULT_INPUT_DIR,
                    help=f"folder of episode .txt screenplays (auto-discovered: {config.DEFAULT_INPUT_DIR})")
    ap.add_argument("--output_dir", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    ap.add_argument("--provider", default="heuristic", choices=["heuristic", "gemini", "openai", "anthropic"],
                    help="stage-2 re-ranker: heuristic = keyword (free, default); others re-rank the shortlist with a "
                         "model. openai also covers any OpenAI-compatible endpoint via OPENAI_BASE_URL")
    ap.add_argument("--model", default=None, help="model id for the provider (ignored by heuristic)")
    ap.add_argument("--api_key", default=None, help="optional; otherwise read from .env / environment")
    ap.add_argument("--min_duration", type=float, default=config.MIN_DURATION)
    ap.add_argument("--max_duration", type=float, default=config.MAX_DURATION)
    ap.add_argument("--shortlist", type=int, default=config.SHORTLIST_SIZE,
                    help="how many stage-1 promos the re-ranker sees")
    ap.add_argument("--time_budget", type=float, default=config.RERANK_TIME_BUDGET_SEC,
                    help="seconds the model re-rank may take; groups past it keep the keyword order (demo-safe)")
    ap.add_argument("--pacing", type=float, default=config.LLM_PACING_SEC,
                    help="seconds between model calls (free-tier rate limits)")
    ap.add_argument("--no_cache", action="store_true", help="ignore cached model answers in .llm_cache/")
    ap.add_argument("--cache_only", action="store_true",
                    help="replay saved model answers only (no key, no network); uncached groups keep the keyword order")
    ap.add_argument("--episodes", default=None, help="subset, e.g. '1-10' or '1,5,12'")
    ap.add_argument("--weights", default=None, help='JSON override, e.g. \'{"hook_strength":0.35}\'')
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def parse_episode_filter(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.input_dir is None or not Path(args.input_dir).is_dir():
        print("ERROR: no episode folder found. Pass --input_dir <folder with episode .txt files>.")
        return 2
    weights = dict(config.DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.loads(args.weights))
    if args.provider == "heuristic" and args.model:
        print(f"note: --model {args.model} is ignored by the heuristic provider (use --provider gemini/openai/anthropic)")
    say = (lambda m: None) if args.quiet else (lambda m: print(m, flush=True))
    try:
        res = run_folder(args.input_dir, args.output_dir, parse_episode_filter(args.episodes), progress=say,
                         provider=args.provider, model=args.model if args.provider != "heuristic" else None,
                         api_key=args.api_key, min_duration=args.min_duration, max_duration=args.max_duration,
                         weights=weights, pacing_sec=args.pacing, use_cache=not args.no_cache,
                         cache_only=args.cache_only, shortlist_size=args.shortlist, time_budget=args.time_budget)
    except LLMError as exc:
        print(f"\nERROR ({type(exc).__name__}): {exc}\nNo outputs were written; raw_outputs/ is unchanged. "
              "Completed calls are cached in .llm_cache/, so a re-run resumes without re-paying for them.")
        return 3
    m = res.meta
    ranked = sorted(res.promos, key=lambda p: p.post_order)
    print("\n" + "=" * 118)
    print(f" {m['show_title']}  |  {m['episodes_parsed']} episodes, {m['corpus_runtime_min']:.1f} min, "
          f"{m['corpus_blocks']} blocks  |  re-ranker {m['provider']}")
    print("=" * 118)
    print(f"{'Post':>4} {'List':<6} {'Promo ID':<26} {'Type':<20} {'Dur':>6} {'Score':>6}  Segments")
    print("-" * 118)
    for p in ranked[:15]:
        print(f"{p.post_order:>4} {('first', 'next', 'later', 'hold')[p.wave - 1]:<6} {p.promo_id:<26} {p.promo_type:<20} "
              f"{p.total_duration_sec:>6.2f} {p.composite_score:>6.2f}  {segments_str(p)}")
    print("-" * 118)
    sb = m["show_brief"]
    print(f"Show brief ({sb.get('source', '?')}): {sb.get('premise_setup', '')} -> {sb.get('premise_twist', '')} | "
          f"stakes words: {', '.join(sb.get('stakes_terms', [])[:10])}")
    print(f"Stage 1:        {m['candidates']} candidates cut, {m['starts_tightened']} starts trimmed to "
          f"{m['pre_roll_sec']}s before the hook, {m['shortlisted']} shortlisted, "
          f"{m['premise_candidates']} sell the premise")
    s2 = m["stage2"]
    if m["reranker"] == "keyword":
        print("Stage 2:        keyword re-ranker (free; no model call)")
    else:
        print(f"Stage 2:        {m['reranker']}: {s2.get('groups', 0)} groups, {s2.get('cached_groups', 0)} from cache, "
              f"{s2.get('fallback_groups', 0)} kept keyword order; why lines kept {s2.get('lines_kept', 0)}, "
              f"rejected {s2.get('lines_rejected', 0)}")
        for r in s2.get("fallback_reasons", [])[:3]:
            print(f"                fallback: {r}")
    print("Plan:           " + " | ".join(f"{k}: {v}" for k, v in m["waves"].items()))
    rk = m.get("ranking", {})
    print(f"                held: {rk.get('held_no_hook', 0)} no hook, {rk.get('held_resolved', 0)} resolved ending, "
          f"{rk.get('held_unclear', 0)} unclear, {rk.get('held_low_score', 0)} low score, "
          f"{rk.get('held_not_shortlisted', 0)} not shortlisted, {rk.get('alternates', 0)} alternates, "
          f"{rk.get('held_plan_full', 0)} plan full")
    print(f"                premise pick in Post first: {rk.get('premise_pick') or 'none'} | finale starts at "
          f"Ep {rk.get('finale_first_episode') or '-'}; {rk.get('finale_to_later', 0)} finale clips moved to Later")
    if m.get("cache_only_preview"):
        print(f"CACHE-ONLY PREVIEW: {m['cache_only_preview']['fallback_groups']} groups had no saved model answer and "
              "kept the keyword order. Not a real run - re-run with network access.")
    print("Promo types:    " + " | ".join(f"{k}: {v}" for k, v in m["promo_types"].items()))
    a = res.audit
    print("\nQuality audit:")
    for c in a["failure_modes"]:
        print(f"  [{c['case_id']}] {c['failure_mode']}: {c['title']}")
    print(f"  flags: {a['summary']['flag_counts']}")
    b = m["llm_budget"]
    u = m.get("llm_usage")
    if u:
        print(f"\nLLM usage ({m['provider']}): {u['calls']} API calls, {u.get('cache_hits', 0)} cache hits | "
              f"{u['input_tokens']:,} input / {u['output_tokens']:,} output / {u['thinking_tokens']:,} thinking tokens | "
              f"${u['cost_usd']:.4f} at list price | {u['rate_limit_retries']} rate-limit retries | "
              f"{u['latency_s']:.0f}s model latency")
    elif b:
        print(f"\nModel re-rank budget for this show: {b['calls']} calls, ~{b['est_input_tokens']:,} in / "
              f"~{b['est_output_tokens']:,} out tokens")
        print("  est. cost per show: " + ", ".join(f"{k} ${v:.3f}" for k, v in b["cost_estimates_usd"].items()))
    lat = m["latency_sec"]
    print(f"\nLatency: parse {lat.get('parse', 0)}s | stage 1 {lat['stage1']}s | stage 2 {lat['stage2']}s | "
          f"rank {lat['rank']}s | audit {lat['audit']}s | TOTAL {lat['total']}s")
    v = res.verification
    print("\nVerification: " + ("ALL CHECKS PASSED" if v.get("all_pass") else "FAILURES"))
    for k, c in v.get("checks", {}).items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}: {c['detail']}")
    print(f"\nOutputs written to {Path(args.output_dir).resolve()}")
    return 0 if v.get("all_pass", False) else 1


if __name__ == "__main__":
    sys.exit(main())
