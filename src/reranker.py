"""Stage 2 - re-rank the shortlist and write the "why it works" line (swappable: keyword or any model).

Stage 1 (code) has already cut every candidate on real screenplay blocks with a 0.3 s pre-roll, rated it blind, held
anything that fails the brief's three tests, and shortlisted the best ~100. Stage 2 only ORDERS that shortlist and
may rewrite one line per promo. It can never change a timestamp, a quote, or let a failing clip back in.

  KeywordReranker  free, offline, deterministic. Order = the stage-1 score; the line = the stake-driven line code
                   already wrote from the clip.
  ModelReranker    any provider in llm_adapter: Gemini, OpenAI or any OpenAI-compatible endpoint (OPENAI_BASE_URL),
                   Anthropic. Clips go in groups of 10, each group spread across the whole score range so scores stay
                   comparable between groups. The model scores each clip 0-100 and writes the line. Code checks every
                   line (any quote must be spoken in the clip; no stock phrases; no invented questions) and keeps the
                   code line when a check fails.
                   Demo-safe: answers are cached per group (a re-run is instant and needs no network), a failing or
                   slow group falls back to the keyword order for that group only, and a time budget caps the stage.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import config
from .evaluator import STOCK_RX, clean_model_copy, strip_invented_questions
from .llm_adapter import LLMAdapter, LLMError, estimate_tokens, price_usd
from .schema import Promo

RERANK_SYSTEM = """You are the performance-marketing editor choosing which vertical microdrama clips to run as paid \
ads (TikTok / Reels / Shorts, ReelShort and DramaBox style).
SHOW: {title}
PREMISE: {setup}
TWIST THE TITLE PROMISES: {twist}
THEMES: {themes}

You get several candidate clips, each headed "### CLIP <id>". You see only what a stranger sees and hears in the clip; \
speakers are anonymised as SPEAKER A, SPEAKER B. Every clip has already passed basic checks (a line or an action in the \
first 3 seconds, an ending that is left open, 20-60 seconds).

For EACH clip return:
  score  0-100: how likely a stranger scrolling past stops in the first 3 seconds AND then wants to start the show. \
Reward a shocking first line or a slap, clear stakes (who wants what, what is at risk), an unanswered ending, and clips \
that sell the twist the title promises. Penalise confusing jumps, talky middles, endings that settle the scene, and \
spoilers beyond what the title already says. Use the full range: in each group the best clip should score at least 30 \
points above the worst.
  why    ONE punchy line, at most 20 words: what is at stake and what is left unanswered. Refer to people as SPEAKER A \
etc. Quote only words actually spoken in the clip. No questions, no hashtags, no marketing cliches.
  sells_premise  true only if the clip shows or teases the twist the title promises.

Return ONLY JSON: {{"clips": [{{"id": "C01", "score": 0, "why": "...", "sells_premise": false}}]}}"""

QUOTE_RX = re.compile(r"[“\"]([^”\"]{3,})[”\"]")


@dataclass
class RerankResult:
    score: float
    why: str
    by: str
    sells_premise: bool = False
    note: str = ""


def check_line(line: str, spoken: Sequence[str]) -> Tuple[bool, str]:
    """A model line is kept only if every quote in it is spoken in the clip, it has no stock phrase, no question
    and a sane length."""
    if not line or len(line) < 15 or len(line) > 170:
        return False, "length"
    if STOCK_RX.search(line):
        return False, f"stock phrase “{STOCK_RX.search(line).group(0)}”"
    outside = QUOTE_RX.sub("", line)
    if "?" in outside:
        return False, "asks a question the clip does not"
    said = " ".join(spoken).lower().replace("’", "'")
    said_n = " ".join(re.findall(r"[a-z0-9']+", said))
    for q in QUOTE_RX.findall(line):
        qn = " ".join(re.findall(r"[a-z0-9']+", q.lower().replace("’", "'")))
        if qn and qn not in said_n:
            return False, f"quote not in the clip (“{q[:40]}”)"
    return True, ""


class KeywordReranker:
    label = "keyword"

    def rerank(self, shortlist: Sequence[Promo], progress: Optional[Callable[[str], None]] = None
               ) -> Dict[str, RerankResult]:
        return {p.promo_id: RerankResult(score=float((p.signals or {}).get("code_score", p.composite_score)),
                                         why=p.why_it_works, by=self.label) for p in shortlist}


class ModelReranker:
    def __init__(self, adapter: LLMAdapter, evaluator: Any, brief: Any, use_cache: bool = True,
                 group_size: int = config.RERANK_GROUP_SIZE, time_budget: float = config.RERANK_TIME_BUDGET_SEC):
        self.adapter, self.ev, self.brief = adapter, evaluator, brief
        self.use_cache, self.group_size, self.time_budget = use_cache, max(2, group_size), time_budget
        self.label = adapter.label
        self.stats: Dict[str, Any] = {"groups": 0, "cached_groups": 0, "fallback_groups": 0, "lines_kept": 0,
                                      "lines_rejected": 0, "fallback_reasons": []}

    # ------------------------------------------------------------------ prompts
    def system(self) -> str:
        b = self.brief
        return RERANK_SYSTEM.format(title=getattr(b, "title", ""), setup=getattr(b, "premise_setup", "") or "-",
                                    twist=getattr(b, "premise_twist", "") or "none",
                                    themes=", ".join(getattr(b, "themes", []) or []) or "-")

    def groups(self, shortlist: Sequence[Promo]) -> List[List[Promo]]:
        """Interleave by stage-1 score so each group spans the whole range (keeps 0-100 comparable across groups)."""
        ranked = sorted(shortlist, key=lambda p: (-(p.signals or {}).get("code_score", 0.0), p.promo_id))
        n = max(1, math.ceil(len(ranked) / self.group_size))
        return [ranked[g::n] for g in range(n)]

    def render(self, group: Sequence[Promo]) -> Tuple[str, Dict[str, Promo]]:
        ids = {f"C{i + 1:02d}": p for i, p in enumerate(group)}
        parts = []
        for cid, p in ids.items():
            cand = self.ev.cands[p.promo_id]
            body = self.ev.render_prompt(cand, self.ev.ctxs.get(p.promo_id), for_batch=True)
            parts.append(f"### CLIP {cid}\n{body}")
        return "\n\n".join(parts) + "\n\nReturn the JSON object only.", ids

    def budget(self, shortlist: Sequence[Promo]) -> Dict[str, Any]:
        sys_t = estimate_tokens(self.system())
        groups = self.groups(shortlist)
        tin = sum(sys_t + estimate_tokens(self.render(g)[0]) for g in groups)
        tout = 60 * len(shortlist)
        models = ("gemini-3.5-flash-lite", "gemini-2.5-flash", "gpt-4o-mini", "claude-haiku-5-5")
        return {"calls": len(groups) + 1, "est_input_tokens": tin, "est_output_tokens": tout,
                "cost_estimates_usd": {m: round(price_usd(m, tin, tout), 4) for m in models}}

    # ------------------------------------------------------------------ run
    def rerank(self, shortlist: Sequence[Promo], progress: Optional[Callable[[str], None]] = None
               ) -> Dict[str, RerankResult]:
        say = progress or (lambda m: None)
        keyword = KeywordReranker().rerank(shortlist)
        out: Dict[str, RerankResult] = {}
        groups = self.groups(shortlist)
        system = self.system()
        t0 = time.time()
        say(f"[stage 2] {self.label} re-ranks {len(shortlist)} shortlisted promos in {len(groups)} groups "
            f"(cached answers are reused; time budget {self.time_budget:.0f} s)")
        for gi, group in enumerate(groups, 1):
            user, ids = self.render(group)
            self.stats["groups"] += 1
            if time.time() - t0 > self.time_budget:
                self._fallback(group, keyword, out, "time budget used up")
                continue
            try:
                data, hit = self.adapter.complete_json_cached(system, user, config.LLM_MAX_OUTPUT_TOKENS,
                                                              tag=f"rerank {gi}/{len(groups)}", use_cache=self.use_cache)
            except LLMError as exc:
                self._fallback(group, keyword, out, str(exc)[:160])
                say(f"  group {gi}/{len(groups)}: keyword fallback ({str(exc)[:90]})")
                continue
            self.stats["cached_groups"] += int(hit)
            rows = {str(r.get("id", "")).strip(): r for r in (data.get("clips") or []) if isinstance(r, dict)}
            for cid, p in ids.items():
                r = rows.get(cid)
                try:
                    score = max(0.0, min(100.0, float(r["score"])))
                except (TypeError, KeyError, ValueError):
                    out[p.promo_id] = RerankResult(keyword[p.promo_id].score, p.why_it_works, "keyword",
                                                   note="model gave no score for this clip")
                    continue
                names = (p.signals or {}).get("speaker_names", {})
                line = strip_invented_questions(clean_model_copy(str(r.get("why", "")), names)).strip()
                ok, why_not = check_line(line, (p.signals or {}).get("clip_lines", []))
                if ok:
                    self.stats["lines_kept"] += 1
                else:
                    self.stats["lines_rejected"] += 1
                out[p.promo_id] = RerankResult(score=round(score, 2), why=line if ok else p.why_it_works,
                                               by=self.label, sells_premise=bool(r.get("sells_premise")),
                                               note="" if ok else f"model line rejected: {why_not}")
            say(f"  group {gi}/{len(groups)}: {'cached' if hit else 'scored'}")
        return out

    def _fallback(self, group: Sequence[Promo], keyword: Dict[str, RerankResult], out: Dict[str, RerankResult],
                  reason: str) -> None:
        self.stats["fallback_groups"] += 1
        if reason not in self.stats["fallback_reasons"]:
            self.stats["fallback_reasons"].append(reason)
        for p in group:
            r = keyword[p.promo_id]
            out[p.promo_id] = RerankResult(r.score, r.why, "keyword", note=f"fallback: {reason}")


def make_reranker(adapter: LLMAdapter, evaluator: Any, brief: Any, use_cache: bool = True):
    return ModelReranker(adapter, evaluator, brief, use_cache) if adapter.is_live else KeywordReranker()
