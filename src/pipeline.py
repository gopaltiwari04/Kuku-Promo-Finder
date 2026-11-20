"""End-to-end orchestration (used by the CLI and by the Streamlit live runner)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import config
from .candidate_generator import CandidateGenerator
from .evaluator import PromoEvaluator
from .lexicons import set_show_terms
from .llm_adapter import LLMAdapter
from .parser import load_corpus, parse_text, resolve_entities
from .quality_auditor import QualityAuditor
from .ranker import rank, shortlist
from .reranker import ModelReranker, make_reranker
from .show_brief import build_brief
from .reporting import episode_stats, sha256, verify, write_csv, write_json, write_report_md
from .schema import Episode, Promo
from .show_profile import profile_for
from .scoring import build_segment, cliff_profile


@dataclass
class PipelineResult:
    promos: List[Promo]
    episodes: List[Episode]
    audit: Dict[str, Any]
    meta: Dict[str, Any]
    stats_rows: List[Dict[str, Any]]
    generator: CandidateGenerator
    verification: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)


def run_on_episodes(episodes: Sequence[Episode], show_title: str, provider: str = "heuristic",
                    model: Optional[str] = None, api_key: Optional[str] = None,
                    min_duration: float = config.MIN_DURATION, max_duration: float = config.MAX_DURATION,
                    weights: Optional[Dict[str, float]] = None, pacing_sec: float = config.LLM_PACING_SEC,
                    use_cache: bool = True, progress: Optional[Callable[[str], None]] = None,
                    errors: Sequence[Dict[str, str]] = (), cache_only: bool = False,
                    shortlist_size: int = config.SHORTLIST_SIZE,
                    time_budget: float = config.RERANK_TIME_BUDGET_SEC) -> PipelineResult:
    """Two stages. Stage 1 (code, deterministic): learn the show, cut and rate every candidate, hold the ones that
    fail the brief's tests, shortlist the best ~100. Stage 2 (keyword or any model): re-rank the shortlist and write
    the why-it-works line. Code then builds the posting plan."""
    say = progress or (lambda m: None)
    t = {"start": time.time()}
    weights = weights or dict(config.DEFAULT_WEIGHTS)
    # explicit provider + missing key raises MissingAPIKeyError here (no silent fallback)
    adapter = LLMAdapter(provider, model, api_key, pacing_sec=pacing_sec, log=say, cache_only=cache_only)
    profile = profile_for(episodes, show_title, adapter)
    brief = build_brief(episodes, show_title, adapter, use_cache, log=say)
    set_show_terms(brief.stakes_terms, list(brief.twist_terms) + list(brief.setup_terms))
    say(f"[2/6] Show: lead {', '.join(sorted(profile.protagonists)) or '-'}; premise \u201c{brief.premise_setup}\u201d -> "
        f"\u201c{brief.premise_twist or '-'}\u201d; twist words {', '.join(brief.twist_terms) or '-'}; "
        f"{len(brief.stakes_terms)} stakes words ({brief.source})")
    say(f"[3/6] Stage 1: cutting candidates for {len(episodes)} episode(s) ...")
    gen = CandidateGenerator(episodes, show_title, min_duration, max_duration, weights, profile=profile, brief=brief)
    cands = gen.generate_all()
    tightened = gen.tighten(cands)
    closing = 0                      # guardrail evidence: cut points whose ending resolves the scene
    for idx in gen.index.values():
        for t_end, _ in idx.clean_ends:
            closing += cliff_profile(build_segment(idx, max(0.0, t_end - 12.0), t_end)).resolved
    t["generate"] = time.time()
    ev = PromoEvaluator(gen, show_title, brief)
    promos = ev.evaluate(cands)
    short = shortlist(promos, weights, size=shortlist_size)
    say(f"[3/6] Stage 1: {len(cands)} candidates rated blind; {len(short)} pass the tests and are shortlisted; "
        f"{tightened} starts tightened to {config.PRE_ROLL_SEC:.1f} s before the first line or action")
    t["stage1"] = time.time()
    rr = make_reranker(adapter, ev, brief, use_cache)
    if hasattr(rr, "time_budget"):
        rr.time_budget = time_budget
    say(f"[4/6] Stage 2: re-ranking the shortlist ({rr.label}) ...")
    results = rr.rerank(short, progress=say)
    for p in short:
        r = results.get(p.promo_id)
        if r is None:
            continue
        p.signals["rerank_score"] = r.score
        p.signals["reranked_by"] = r.by
        if r.sells_premise:
            p.signals["model_sells_premise"] = True
        if r.note:
            p.signals["rerank_note"] = r.note
        if r.why and r.why != p.why_it_works:
            p.signals["code_why"] = p.why_it_works
            p.why_it_works = r.why
    t["stage2"] = time.time()
    say("[5/6] Posting plan: Post first / Post next / Later ...")
    promos = rank(promos, weights)
    rank_info = dict(rank.last_info)
    t["rank"] = time.time()
    say("[6/6] Quality audit ...")
    audit = QualityAuditor(gen, promos).run()
    t["audit"] = time.time()
    corpus_chars = sum(len(Path(e.path).read_text(encoding="utf-8", errors="ignore")) if Path(e.path).exists() else 0
                       for e in episodes)
    mr = rr if isinstance(rr, ModelReranker) else ModelReranker(adapter, ev, brief)   # prompts shown/estimated either way
    budget = mr.budget(short) if short else {}
    meta = {
        "show_title": show_title, "provider": adapter.label, "requested_provider": adapter.requested_provider,
        "reranker": rr.label, "episodes_parsed": len(episodes),
        "parse_errors": list(errors), "corpus_runtime_min": round(sum(e.duration_sec for e in episodes) / 60, 2),
        "corpus_blocks": sum(len(e.events) for e in episodes), "corpus_chars": corpus_chars,
        "corpus_tokens_est": int(corpus_chars / config.CHARS_PER_TOKEN),
        "windows_evaluated": gen.stats.windows_evaluated, "clean_starts": gen.stats.starts_total,
        "starts_passing_hook_gate": gen.stats.starts_passing_gate, "clean_ends": gen.stats.ends_total,
        "candidates": len(cands), "promos": len(promos), "shortlisted": len(short), "starts_tightened": tightened,
        "pre_roll_sec": config.PRE_ROLL_SEC,
        "promo_types": {k: sum(p.promo_type == k for p in promos) for k in
                        ("continuous", "multi_segment_intra", "multi_segment_cross")},
        "premise_candidates": sum(bool(p.signals.get("premise_reveal")) for p in promos),
        "waves": {config.WAVE_LABELS[k]: sum(p.wave == k for p in promos) for k in (1, 2, 3, 4)},
        "ranking": rank_info,
        "min_duration": min_duration, "max_duration": max_duration, "weights": weights,
        "show_brief": brief.to_dict(),
        "stage2": dict(getattr(rr, "stats", {}), reranker=rr.label),
        "llm_budget": budget,
        "llm_usage": dict(vars(adapter.meter), pacing_sec=adapter.pacing_sec,
                          backoff_schedule_sec=adapter.backoff_schedule) if adapter.is_live else None,
        "latency_sec": {"stage1": round(t["stage1"] - t["start"], 2), "stage2": round(t["stage2"] - t["stage1"], 2),
                        "rank": round(t["rank"] - t["stage2"], 3), "audit": round(t["audit"] - t["rank"], 2),
                        "total": round(t["audit"] - t["start"], 2)},
        "sample_llm_prompts": ([{"batch": 1, "system": mr.system(), "user": mr.render(mr.groups(short)[0])[0]}]
                               if short else []),
        "show_profile": profile.to_dict(),
        "cache_only_preview": ({"fallback_groups": getattr(rr, "stats", {}).get("fallback_groups", 0)}
                               if adapter.cache_only else None),
        "guardrail_stats": {"closing_endings": int(closing), "clean_ends": sum(len(i.clean_ends) for i in gen.index.values())},
        "story_sequences": sequence_spans(episodes, gen.sequences),
    }
    stats_rows = episode_stats(episodes, gen.index, promos, gen.stats.per_episode_windows)
    return PipelineResult(promos=promos, episodes=list(episodes), audit=audit, meta=meta, stats_rows=stats_rows,
                          generator=gen)


def sequence_spans(episodes: Sequence[Episode], sequences: Dict[Any, str]) -> Dict[str, List[List[float]]]:
    """Story sequence -> the scene time spans it covers ([episode, start_sec, end_sec]); timings only, no script text.
    Lets quality_eval measure repetition for any list of clips, including plans from older runs."""
    out: Dict[str, List[List[float]]] = {}
    for ep in episodes:
        for sc in ep.scenes:
            sid = sequences.get((ep.episode_number, sc.scene_id))
            if sid:
                out.setdefault(sid, []).append([ep.episode_number, round(sc.start_sec, 2), round(sc.end_sec, 2)])
    return out


def run_folder(input_dir: Path, output_dir: Optional[Path] = None, episodes_filter: Optional[Sequence[int]] = None,
               progress: Optional[Callable[[str], None]] = None, **kw) -> PipelineResult:
    say = progress or (lambda m: None)
    t0 = time.time()
    input_dir = Path(input_dir)
    files = [p for p in input_dir.iterdir() if p.is_file() and config._looks_like_episode(p)]
    hashes = {str(p): sha256(p) for p in files}
    say(f"[1/6] Parsing {len(files)} screenplay(s) from {input_dir} ...")
    episodes, errors = load_corpus(input_dir)
    if episodes_filter:
        keep = set(episodes_filter)
        episodes = [e for e in episodes if e.episode_number in keep]
    t_parse = time.time() - t0
    res = run_on_episodes(episodes, config.show_title_from_dir(input_dir), progress=progress, errors=errors, **kw)
    res.meta["latency_sec"]["parse"] = round(t_parse, 2)
    res.meta["latency_sec"]["total"] = round(res.meta["latency_sec"]["total"] + t_parse, 2)
    try:
        res.meta["input_dir"] = str(input_dir.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        res.meta["input_dir"] = str(input_dir)
    if output_dir is not None:
        write_outputs(res, Path(output_dir))
        n_files = len(episodes) if episodes_filter else len(files)
        res.verification = verify(res.promos, res.episodes, errors, n_files, Path(output_dir) / "ranked_promos.csv",
                                  Path(output_dir) / "promos_all_episodes.json", hashes,
                                  kw.get("min_duration", config.MIN_DURATION), kw.get("max_duration", config.MAX_DURATION),
                                  generator=res.generator, provider_label=res.meta["provider"],
                                  cache_only=bool(res.meta.get("cache_only_preview")))
        write_json(Path(output_dir) / "verification_report.json", res.verification)
        from .quality_eval import evaluate_run
        res.quality = evaluate_run(Path(output_dir))
    return res


def run_texts(texts: Sequence[str], names: Sequence[str], show_title: str = "Uploaded screenplay",
              progress: Optional[Callable[[str], None]] = None, **kw) -> PipelineResult:
    """Live interview path: raw screenplay text(s) pasted/uploaded in the UI."""
    episodes = []
    for i, (txt, nm) in enumerate(zip(texts, names), 1):
        ep = parse_text(txt, filename=nm, path="")
        if not ep.episode_number:
            ep.episode_number = i
        episodes.append(ep)
    nums = [e.episode_number for e in episodes]
    if len(set(nums)) != len(nums):
        for i, e in enumerate(episodes, 1):
            e.episode_number = i
    resolve_entities(episodes)
    return run_on_episodes(episodes, show_title, progress=progress, **kw)


# Saved outputs keep as little screenplay text as the deliverable needs (the hook and cut lines); the input pack is
# confidential and the repository is public.
EVIDENCE_LINES, EVIDENCE_CHARS = 6, 160


def _public_promo(p: Promo) -> Dict[str, Any]:
    d = p.to_dict()
    d.get("signals", {}).pop("hook_window_lines", None)
    d.get("signals", {}).pop("clip_lines", None)
    return d


def _public_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(meta)
    m["sample_llm_prompts"] = [{"batch": s.get("batch"), "system": s.get("system"),
                                "user_prompt_chars": len(s.get("user", ""))} for s in meta.get("sample_llm_prompts", [])]
    return m


def _public_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    a = dict(audit)
    a["failure_modes"] = [dict(fm, evidence=[str(x)[:EVIDENCE_CHARS] for x in (fm.get("evidence") or [])[:EVIDENCE_LINES]])
                          for fm in audit.get("failure_modes", [])]
    return a


def write_outputs(res: PipelineResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ranked = sorted(res.promos, key=lambda p: p.post_order)
    write_json(out / "promos_all_episodes.json", {
        "metadata": {k: v for k, v in res.meta.items() if k != "sample_llm_prompts"},
        "promos": [_public_promo(p) for p in ranked],
    })
    write_csv(out / "ranked_promos.csv", [p for p in ranked if p.wave in (1, 2, 3)])
    write_csv(out / "held_promos.csv", [p for p in ranked if p.wave == 4])
    write_report_md(out / "ranked_promos_report.md", ranked, res.meta, res.audit)
    import csv
    with open(out / "episode_summary_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res.stats_rows[0].keys()))
        w.writeheader()
        w.writerows(res.stats_rows)
    write_json(out / "quality_audit_and_bad_picks.json", _public_audit(res.audit))
    write_json(out / "run_metadata.json", _public_meta(res.meta))
