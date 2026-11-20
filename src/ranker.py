"""Stage 1 shortlist and the posting plan (100% code, reproducible).

Stage-1 score out of 100 = 10 x (0.30 hook + 0.25 cliffhanger + 0.25 makes-sense-alone + 0.10 pacing
                                 + 0.10 virality - 0.12 spoiler risk)

1. Hold anything that fails one of the brief's three tests:
     no hook   - no line and no physical action in the first 3 s, or a weak one
     resolved  - the ending closes the scene (last line, final shot or score)
     unclear   - makes-sense-alone below the floor
2. shortlist(): the best ~100 that pass go to stage 2, which may re-order them (src/reranker.py). The posting order
   below uses the stage-2 score; the floors above always use the stage-1 (code) score.
3. Walk the shortlist best-first and fill three short lists (config.POSTING_RULES):
     Post first (5)   one slot is reserved for the best promo that sells the title's twist (premise); the rest come
                      from the first quarter of the show with spoiler <= 3.5
     Post next (10)   spoiler <= 5
     Later (up to 25) everything else that passes, after a spoiler check
   Post first and Post next are cold promos (strangers): nothing from the finale (last 10% of episodes). A finale
   clip can only go to Later, marked for retargeting viewers who are already that far in.
   A promo that shares footage with a better pick, starts within 3 s of where one ends, or would be a third post
   from the same story sequence in a list (the four-episode wedding is one sequence) is not posted next to it:
   footage clashes become ALTERNATES (kept for A/B tests), sequence overflow moves down a list.
3. Order inside each list = score order. Nothing else moves a promo up or down.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config
from .candidate_generator import finale_first_episode
from .schema import Promo
from .scoring import composite


def _ts(ts: str) -> float:
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def _spans(p: Promo) -> List[Tuple[int, float, float]]:
    return [(s["episode"], _ts(s["start_timestamp"]), _ts(s["end_timestamp"])) for s in p.segments]


def clash(a: Promo, b: Promo, chain_gap: float) -> Optional[str]:
    """Why b cannot be posted next to a: they share footage, or one starts where the other ends."""
    for ep1, s1, e1 in _spans(a):
        for ep2, s2, e2 in _spans(b):
            if ep1 != ep2:
                continue
            overlap = min(e1, e2) - max(s1, s2)
            if overlap > 0.5:
                return f"shares {overlap:.1f} s of footage with {a.promo_id}"
            if -1e-6 <= s2 - e1 <= chain_gap or -1e-6 <= s1 - e2 <= chain_gap:
                return f"starts where {a.promo_id} ends (one scene cut in two)"
    return None


def first_quarter_episode(promos: Sequence[Promo], share: float) -> int:
    eps = sorted({e for p in promos for e in p.episodes})
    if not eps:
        return 1
    n = eps[-1] - eps[0] + 1
    return eps[0] - 1 + max(1, math.ceil(share * n))


def gate(p: Promo, rules: Dict[str, float]) -> Optional[str]:
    """The brief's three tests, plus a minimum score. Returns the hold reason or None."""
    sig = p.signals or {}
    if sig.get("hook_pass") is False:
        why = "; ".join(sig.get("hook_gate_reasons") or []) or "no line or action in the first 3 s"
        return f"no hook: {why}"
    if p.scores["hook_strength"] < rules["min_hook"]:
        return f"weak hook: the first 3 s rate {p.scores['hook_strength']:.1f} < {rules['min_hook']}"
    if sig.get("resolved"):
        return "ending resolves the scene (" + "; ".join(sig.get("resolution_reasons") or [])[:160] + ")"
    if p.scores["standalone_clarity"] < rules["min_clarity"]:
        return (f"a stranger would struggle to follow it (makes-sense-alone {p.scores['standalone_clarity']:.1f} "
                f"< {rules['min_clarity']})")
    code = sig.get("code_score", p.composite_score)
    if code < rules["min_score"]:
        return f"score {code:.1f} < {rules['min_score']:.0f}"
    return None


def shortlist(promos: Sequence[Promo], weights: Optional[Dict[str, float]] = None,
              rules: Optional[Dict[str, float]] = None, size: int = config.SHORTLIST_SIZE) -> List[Promo]:
    """Stage 1: score every promo with code, keep the best `size` that pass the three tests for stage 2. A promo
    that sells the title's twist is always kept if it passes, so stage 2 can never drop the premise."""
    weights = weights or config.DEFAULT_WEIGHTS
    rules = rules or config.POSTING_RULES
    for p in promos:
        p.signals["code_score"] = round(composite(p.scores, weights), 2)
        p.composite_score = p.signals["code_score"]
    ok = sorted([p for p in promos if gate(p, rules) is None], key=lambda x: (-x.signals["code_score"], x.promo_id))
    keep = ok[:size]
    for p in ok[size:]:
        if p.signals.get("premise_reveal"):
            keep.append(p)
    ids = {p.promo_id for p in keep}
    for i, p in enumerate(ok, 1):
        p.signals["stage1_rank"] = i
    for p in promos:
        p.signals["shortlisted"] = p.promo_id in ids
    return keep


def _seq_full(p: Promo, members: Sequence[Promo], cap: int) -> bool:
    seqs = (p.signals or {}).get("sequences") or []
    return any(sum(1 for q in members if sq in ((q.signals or {}).get("sequences") or [])) >= cap for sq in seqs)


def assign_waves(promos: Sequence[Promo], rules: Dict[str, float]) -> Dict[str, Any]:
    by_score = sorted(promos, key=lambda x: (-x.composite_score, x.promo_id))
    first_ep = first_quarter_episode(promos, rules["first_quarter"])
    finale_from = finale_first_episode(sorted({e for p in promos for e in p.episodes}))
    chain_gap = float(rules["chain_gap_sec"])
    cap = int(rules["max_per_sequence"])
    lists: Dict[int, List[Promo]] = {1: [], 2: [], 3: []}
    size = {1: int(rules["post_first"]), 2: int(rules["post_next"]), 3: int(rules["later"])}
    placed: List[Promo] = []
    premise_used = 0
    counts = {"held_no_hook": 0, "held_resolved": 0, "held_unclear": 0, "held_low_score": 0, "alternates": 0,
              "held_plan_full": 0, "held_not_shortlisted": 0, "finale_to_later": 0}
    for p in by_score:
        p.wave, p.variant_of, p.post_reason, p.wave_reason = 4, "", "", ""
        p.review_reasons = [r for r in p.review_reasons if not r.startswith(("title reveal", "spoiler ", "retarget"))]
        (p.signals or {}).pop("audience", None)
    # the premise slot: the best shortlisted promo that sells the title's twist goes into Post first
    premise_pick = next((p for p in by_score if (p.signals or {}).get("premise_reveal") and gate(p, rules) is None
                         and (p.signals or {}).get("shortlisted", True) and max(p.episodes) < finale_from), None)
    if premise_pick is not None and int(rules["premise_slots"]) > 0:
        premise_pick.wave = 1
        premise_pick.wave_reason = (f"premise slot: it sells the title's twist (Ep {max(premise_pick.episodes)}); the "
                                    "title already promises it, so it is a hook, not a spoiler")
        premise_pick.review_reasons.append("title reveal: confirm the clip gives away nothing beyond the title")
        lists[1].append(premise_pick)
        placed.append(premise_pick)
        premise_used = 1
    for p in by_score:
        if p is premise_pick:
            continue
        hold = gate(p, rules)
        if hold:
            p.wave_reason = "held: " + hold
            key = ("held_no_hook" if hold.startswith(("no hook", "weak hook")) else "held_resolved" if hold.startswith("ending")
                   else "held_unclear" if hold.startswith("a stranger") else "held_low_score")
            counts[key] += 1
            continue
        if not (p.signals or {}).get("shortlisted", True):
            p.wave_reason = (f"held: passes every test but ranks {(p.signals or {}).get('stage1_rank', '?')} in stage 1; "
                             f"only the best {config.SHORTLIST_SIZE} go to stage 2")
            counts["held_not_shortlisted"] += 1
            continue
        twin, why = None, ""
        for a in placed:
            why = clash(a, p, chain_gap) or ""
            if why:
                twin = a
                break
        if twin is not None:
            p.variant_of = twin.promo_id
            p.wave_reason = f"alternate of {twin.promo_id}: {why} (keep it for A/B tests, not as a separate post)"
            counts["alternates"] += 1
            continue
        sig = p.signals or {}
        spoiler = p.scores["spoiler_risk"]
        premise = bool(sig.get("premise_reveal"))
        finale = max(p.episodes) >= finale_from

        def seq_full(tier: int) -> bool:
            return _seq_full(p, lists[tier], cap)

        target, reason = None, ""
        early = max(p.episodes) <= first_ep
        if not finale and len(lists[1]) < size[1] and not seq_full(1):
            if early and spoiler <= rules["first_max_spoiler"]:
                target, reason = 1, (f"from the first quarter of the show (Ep 1-{first_ep}), spoiler {spoiler:.1f}")
            elif premise and premise_used < int(rules["premise_slots"]):
                target, reason = 1, (f"title reveal from Ep {max(p.episodes)}: it sells what the show title already "
                                     "promises, so it is a hook, not a spoiler")
                premise_used += 1
                p.review_reasons.append("title reveal: confirm the clip gives away nothing beyond the title")
        if (target is None and not finale and len(lists[2]) < size[2] and spoiler <= rules["next_max_spoiler"]
                and not seq_full(2)):
            not_first = ("Post first is full" if len(lists[1]) >= size[1] else
                         "Post first already holds this scene twice" if seq_full(1) else
                         f"spoiler {spoiler:.1f} > {rules['first_max_spoiler']}" if spoiler > rules["first_max_spoiler"]
                         else f"after Ep {first_ep}")
            target, reason = 2, f"not Post first because {not_first}; spoiler {spoiler:.1f}"
        if target is None and len(lists[3]) < size[3] and not seq_full(3):
            target, reason = 3, (f"finale (Ep {max(p.episodes)} of the last 10%): retarget viewers already past "
                                 f"Ep {finale_from - 1}, never a cold promo" if finale
                                 else f"spoiler {spoiler:.1f}: check it before posting" if spoiler > rules["next_max_spoiler"]
                                 else "passes every test; the lists above are full or already hold this scene twice")
            if finale:
                counts["finale_to_later"] += 1
                p.review_reasons.append(f"retarget only: shows the finale (Ep {max(p.episodes)})")
        if target is None:
            p.wave_reason = ("held: passes every test, but the plan is full or already posts this scene twice in "
                             "every list")
            counts["held_plan_full"] += 1
            continue
        p.wave, p.wave_reason = target, reason
        lists[target].append(p)
        placed.append(p)
        if spoiler > rules["human_spoiler_check_above"]:
            p.review_reasons.append(f"spoiler {spoiler:.1f}: confirm the clip gives nothing away")
    for p in promos:
        p.wave_label = config.WAVE_LABELS[p.wave]
        p.needs_human_review = bool(p.review_reasons)
        if p.wave in (1, 2):
            p.signals["audience"] = "cold"
        elif p.wave == 3:
            p.signals["audience"] = "retarget" if max(p.episodes) >= finale_from else "later"
    return {"first_quarter_last_episode": first_ep, "finale_first_episode": finale_from,
            "premise_pick": premise_pick.promo_id if premise_pick is not None and premise_pick.wave == 1 else None,
            "premise_picks": premise_used,
            "sizes": {config.WAVE_LABELS[k]: len(v) for k, v in lists.items()}, **counts}


def _order(promos: Sequence[Promo]) -> List[Promo]:
    out: List[Promo] = []
    for w in (1, 2, 3):
        grp = sorted([p for p in promos if p.wave == w], key=lambda x: (-x.composite_score, x.promo_id))
        for i, p in enumerate(grp, 1):
            p.post_reason = f"{config.WAVE_LABELS[w]} #{i} by score: {p.wave_reason}"
        out.extend(grp)
    hold = sorted([p for p in promos if p.wave == 4], key=lambda x: (-x.composite_score, x.promo_id))
    for p in hold:
        p.post_reason = "not posted: " + p.wave_reason
    return out + hold


def rank(promos: Sequence[Promo], weights: Optional[Dict[str, float]] = None,
         rules: Optional[Dict[str, float]] = None) -> List[Promo]:
    """Posting order = the stage-2 score for shortlisted promos (the code score when stage 2 is the keyword
    re-ranker or fell back), the code score for everything else."""
    weights = weights or config.DEFAULT_WEIGHTS
    rules = rules or config.POSTING_RULES
    for p in promos:
        code = round(composite(p.scores, weights), 2)
        p.signals["code_score"] = code
        rr = p.signals.get("rerank_score")
        p.composite_score = float(rr) if (rr is not None and p.signals.get("shortlisted")) else code
    for i, p in enumerate(sorted(promos, key=lambda x: (-x.composite_score, x.promo_id)), 1):
        p.score_rank = i
    info = assign_waves(promos, rules)
    posted = _order(promos)
    for i, p in enumerate(posted, 1):
        p.post_order = i
    rank.last_info = info            # read by the pipeline for run_metadata
    return posted


rank.last_info = {}
