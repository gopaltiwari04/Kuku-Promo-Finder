"""Shared deterministic drama scorer over a *blind* context.

Used twice:
  * Stage 2 (candidate_generator) to shortlist windows cheaply, and
  * Stage 3 as the zero-cost `heuristic` provider and as the fallback when an LLM call fails.

It only ever looks at events inside the clip window(s): no logline, no continuity notes,
no speaker names (anchors and vocatives must be *spoken* to count).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import config
from .lexicons import (CLOSURE_TAG_RX, DEPARTURE_RX, FINAL_BEAT_RX, KINETIC_ACTION_RX, LORE_RX, PRONOUN_RX,
                       RESOLUTION_RX, RESOLVE_VISUAL_RX, REVEAL_MARK_RX, TROPE_WEIGHTS, TWIST_RX, emotion_arousal,
                       trope_hits, show_terms_in)
from .parser import EpisodeIndex, EventFeatures
from .schema import ScriptEvent

ACT_CLIFF = {"question": 2.0, "threat": 2.2, "reveal": 2.0, "accusation": 1.4, "vow": 1.7, "command": 1.2,
             "plea": 1.5}
ACT_PRIORITY = ("reveal", "threat", "accusation", "command", "vow", "plea", "question")
CAP_RX = re.compile(r"\b([A-Z][a-z]{2,})\b")


def spoken_names(text: str) -> Set[str]:
    """Capitalised words that are not sentence-initial = probable names / vocatives."""
    out = set()
    for m in CAP_RX.finditer(text):
        pre = text[:m.start()].rstrip(" \"'“”")
        if not pre or pre[-1] in ".!?…":
            continue
        if m.group(1) not in NOT_NAMES and not show_terms_in(m.group(1), "premise"):   # a rank, not a name
            out.add(m.group(1))
    return out
NOT_NAMES = {"The", "Then", "There", "This", "That", "What", "Why", "How", "When", "Where", "Who", "You",
             "Your", "Yes", "Not", "And", "But", "Because", "Please", "Look", "Lady", "Lord", "Sir", "Father",
             "Mother", "Sister", "Brother", "God", "Enough", "Stop", "King", "Queen", "Prince", "Princess", "Boss",
             "Now", "Never", "Long", "Our", "His", "Her", "Their", "Let", "Come", "Mrs", "Mister", "Majesty",
             "Grace", "Highness", "Madam", "Well", "Good", "Every", "Only", "Even", "Here", "Get", "Take",
             "Do", "Don", "Its", "It", "She", "He", "They", "We", "My", "Of", "Oh", "Ah", "No"}


def location_key(heading: str) -> str:
    """'INT. CATHEDRAL AISLE – CONTINUOUS' -> 'INT. CATHEDRAL' (same place for a viewer)."""
    h = re.split(r"\s+[–—-]\s+", heading.strip())[0].upper()
    words = h.split()
    return " ".join(words[:2]) if words else h


def title_tokens(show_title: str, extra: Iterable[str] = ()) -> Set[str]:
    """Lower-case words of the title, plus any learned twist words (the show brief's twist_terms)."""
    return set(re.findall(r"[a-z]+", show_title.lower())) | {w.lower() for w in extra if w}


@dataclass
class BlindSegment:
    episode: int
    start_sec: float
    end_sec: float
    events: List[ScriptEvent]
    feats: Dict[int, EventFeatures]
    headings: List[str] = field(default_factory=list)

    @cached_property
    def dialogue(self) -> List[ScriptEvent]:
        """Spoken lines fully inside the window (clean cut points guarantee no clipped lines)."""
        return [e for e in self.events if e.track == "DIALOGUE"
                and e.start_sec >= self.start_sec - 1e-6 and e.end_sec <= self.end_sec + 1e-6]

    @cached_property
    def visuals(self) -> List[ScriptEvent]:
        return [e for e in self.events if e.track != "DIALOGUE"]

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 2)


@dataclass
class BlindContext:
    segments: List[BlindSegment]

    @property
    def total_duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> List[int]:
        out: List[int] = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out

    def all_dialogue(self) -> List[ScriptEvent]:
        return [d for s in self.segments for d in s.dialogue]

    def all_text(self, include_visual: bool = True) -> str:
        parts = []
        for s in self.segments:
            parts.extend(d.text for d in s.dialogue)
            if include_visual:
                parts.extend(v.text for v in s.visuals)
        return " \n".join(parts)


def build_segment(idx: EpisodeIndex, start: float, end: float) -> BlindSegment:
    evs = idx.events_in(start, end)
    heads = []
    for sid in dict.fromkeys(e.scene_id for e in evs):
        for sc in idx.ep.scenes:
            if sc.scene_id == sid:
                heads.append(sc.heading)
    return BlindSegment(episode=idx.ep.episode_number, start_sec=start, end_sec=end, events=evs,
                        feats=idx.feat, headings=heads)


# ======================================================================================
# Individual score components (0..10)
# ======================================================================================
@dataclass
class HookProfile:
    score: float
    passes_gate: bool
    hook_type: str
    event: Optional[ScriptEvent]
    visual: Optional[ScriptEvent]
    first_dialogue_offset: Optional[float]
    reasons: List[str] = field(default_factory=list)


def hook_profile(seg: BlindSegment, hook_sec: float = config.HOOK_WINDOW_SEC) -> HookProfile:
    s, H = seg.start_sec, seg.start_sec + hook_sec
    dl = seg.dialogue
    in_hook = [d for d in dl if d.start_sec <= H + 1e-6]
    vis = [e for e in seg.visuals if e.start_sec <= H + 1e-6 and e.end_sec > s + 1e-6]
    reasons: List[str] = []
    best_a, best_d = 0.0, None
    for d in in_hook:
        a = seg.feats[d.idx].arousal * (1.0 if d.start_sec - s <= 1.5 else 0.88)
        if a > best_a:
            best_a, best_d = a, d
    kin_ev = max(vis, key=lambda e: seg.feats[e.idx].kinetic, default=None)
    kin = seg.feats[kin_ev.idx].kinetic if kin_ev is not None else 0.0
    first_off = (dl[0].start_sec - s) if dl else None
    action = any((e.track == "SOUND_FX" and seg.feats[e.idx].sfx_impact) or KINETIC_ACTION_RX.search(e.text)
                 for e in vis)
    passes = bool(in_hook) or (kin >= 1.6 and action)
    if not in_hook and kin >= 1.6 and not action:
        reasons.append("visual-only opening shows a picture, not an action")
    raw = 1.4 + 1.15 * best_a + 0.7 * kin
    if not in_hook:
        lead = (first_off if first_off is not None else seg.duration) - hook_sec
        raw -= 0.6 * max(0.0, lead)
        reasons.append(f"no line in first {hook_sec:.0f}s")
    first = in_hook[0] if in_hook else (dl[0] if dl else None)
    if first is not None:
        f = seg.feats[first.idx]
        if f.is_continuation:
            raw -= 3.0
            passes = False
            reasons.append("opens mid-sentence")
        elif f.dependent_opener and first.start_sec <= H:
            raw -= 1.2
            reasons.append("opens on a reply that needs prior context")
    if best_d is not None and seg.feats[best_d.idx].anchors:
        raw += 0.4
    if len(in_hook) >= 2:
        raw += 0.3
    if any(seg.feats[e.idx].closeup for e in vis):
        raw += 0.15
    if best_d is not None and emotion_arousal(best_d.emotion_tags) < 0:
        raw -= 0.8
    # the hook "type" drives the copy
    if best_d is not None and (best_a >= kin * 1.3 or kin < 2.0):
        acts = seg.feats[best_d.idx].acts
        htype = next((a for a in ACT_PRIORITY if a in acts), "line")
        ev = best_d
    elif kin_ev is not None and kin >= 1.6:
        htype, ev = "kinetic", kin_ev
    else:
        htype, ev = "silent", None
    visual = next((e for e in vis if e.track == "CAMERA"), None)
    return HookProfile(score=max(0.0, min(10.0, raw)), passes_gate=passes, hook_type=htype, event=ev,
                       visual=kin_ev if htype == "kinetic" else visual, first_dialogue_offset=first_off,
                       reasons=reasons)


@dataclass
class CliffProfile:
    score: float
    cliff_type: str
    line: Optional[ScriptEvent]
    tail_visual: Optional[ScriptEvent]
    tail_sfx: Optional[ScriptEvent]
    reasons: List[str] = field(default_factory=list)
    closure: int = 0          # how many signs say the scene is closing (>= 2 = the cut resolves, not hangs)
    resolved: bool = False    # the ending resolves the scene (last line or final shots/sound) - not a cliffhanger
    resolution_reasons: List[str] = field(default_factory=list)


# hostile or agitated delivery = the scene is still live (a cruel "Say goodbye to your job" is a threat)
HOSTILE_TAG_RX = re.compile(r"cruel|menac|sinister|threat|malic|venom|furious|angry|enraged|cold|icy|vicious|mock|"
                            r"sneer|taunt|gloat|deadly|chill|ominous|shout|scream|yell|hiss|snarl|warning", re.I)
ULTIMATUM_RX = re.compile(r",\s*or\b|\bor else\b|\bor you\b|\buntil\b|\bbefore (?:dawn|nightfall|sunrise)\b", re.I)
OPEN_ACTS = {"question", "threat", "command", "reveal", "plea", "accusation"}


def closure_signals(seg: BlindSegment, tail_sec: float = config.CLIFF_WINDOW_SEC) -> Tuple[int, List[str]]:
    """Signs that the final beat closes the scene (a resolution) instead of leaving it hanging.
    A line that asks, threatens, orders, reveals, pleads or trails off is open; otherwise count:
    resolving delivery (proud/triumphant/calm...), departure or farewell words, the last line of a monologue
    nobody answers, and a final shot written as an ending ("final proclamation", "head held high")."""
    dl = seg.dialogue
    if not dl:
        return 0, []
    last = dl[-1]
    if seg.end_sec - last.end_sec > tail_sec + 1e-6:
        return 0, []
    f = seg.feats[last.idx]
    text = last.text.strip()
    acts = set(f.acts)
    if f.is_continuation:                       # "...no one will remember" finishes the previous line's sentence
        for d in reversed(dl[:-1]):
            if d.speaker != last.speaker:
                break
            acts |= set(seg.feats[d.idx].acts)
            if not seg.feats[d.idx].is_continuation:
                break
    tags = " ".join(last.emotion_tags)
    if (text.endswith("?") or text.endswith("...") or text.endswith("\u2026") or f.continues or acts & OPEN_ACTS
            or HOSTILE_TAG_RX.search(tags) or ULTIMATUM_RX.search(text)):
        return 0, []
    n, why = 0, []
    if CLOSURE_TAG_RX.search(tags):
        n += 1
        why.append("delivered as a resolution")
    if DEPARTURE_RX.search(text):
        n += 1
        why.append("departure/farewell line")
    run = 0
    for d in reversed(dl):
        if d.speaker != last.speaker:
            break
        run += 1
    if run >= 3:
        n += 1
        why.append(f"ends a {run}-line monologue nobody answers")
    tail_text = " ".join(v.text for v in seg.visuals if v.end_sec > seg.end_sec - tail_sec - 1e-6)
    if FINAL_BEAT_RX.search(tail_text):
        n += 1
        why.append("final shot written as an ending")
    return n, why


HARD_OPEN_ACTS = {"threat", "question", "plea", "reveal"}
OPEN_LINE_TAIL_SEC = 1.5


def visual_resolution(seg: BlindSegment, tail_sec: float = config.CLIFF_WINDOW_SEC) -> List[str]:
    """Signs in the final shots and sound that the scene is resolving: walking off together, a soft smile,
    an embrace, a triumphant score fading out. Overridden when the clip still ends on a live line
    (a threat, question, plea or reveal in the last 1.5 s, or a hostile delivery)."""
    e = seg.end_sec
    tail = [v for v in seg.visuals if v.end_sec > e - tail_sec + 1e-6 and v.start_sec < e - 1e-6]
    why: List[str] = []
    for v in tail:
        if v.track == "SOUND_FX":
            if seg.feats[v.idx].sfx_resolves:
                why.append("score resolves: " + v.text[:60])
            elif seg.feats[v.idx].sfx_cliff or seg.feats[v.idx].sfx_impact:
                return []                      # a sting or impact on the cut keeps it open
        else:
            m = RESOLVE_VISUAL_RX.search(v.text)
            if m:
                why.append("final shot resolves: \u201c" + m.group(0)[:50] + "\u201d")
    if not why:
        return []
    dl = seg.dialogue
    if dl:
        last = dl[-1]
        f = seg.feats[last.idx]
        tender = emotion_arousal(last.emotion_tags) < 0      # a gentle "Are you hurt?" does not keep it open
        if e - last.end_sec <= OPEN_LINE_TAIL_SEC + 1e-6 and not tender and (
                set(f.acts) & HARD_OPEN_ACTS or last.text.rstrip().endswith("?")
                or HOSTILE_TAG_RX.search(" ".join(last.emotion_tags))):
            return []
    return why


def cliff_profile(seg: BlindSegment, tail_sec: float = config.CLIFF_WINDOW_SEC) -> CliffProfile:
    e, F = seg.end_sec, seg.end_sec - tail_sec
    dl = seg.dialogue
    last = dl[-1] if dl else None
    tail = [v for v in seg.visuals if v.end_sec > F + 1e-6 and v.start_sec < e - 1e-6]
    reasons: List[str] = []
    raw = 1.5
    ctype = "visual"
    if last is not None and e - last.end_sec <= tail_sec + 1e-6:
        f = seg.feats[last.idx]
        act_val = max((ACT_CLIFF[a] for a in f.acts), default=0.0)
        val = act_val + 0.55 * f.arousal
        gap = e - last.end_sec
        if gap > 2.0:
            val *= 0.8
        raw += val
        ctype = "question" if "question" in f.acts else next((a for a in ACT_PRIORITY if a in f.acts), "line")
        if emotion_arousal(last.emotion_tags) < 0 and act_val < 1.5:
            raw -= 1.8
            reasons.append("final line is tender/resolving")
        if f.continues:
            raw -= 3.0
            reasons.append("cuts mid-sentence")
    else:
        reasons.append("no line in final beat")
    closeup = next((v for v in tail if seg.feats[v.idx].closeup), None)
    sfx_cliff = next((v for v in tail if v.track == "SOUND_FX" and seg.feats[v.idx].sfx_cliff), None)
    sfx_imp = next((v for v in tail if v.track == "SOUND_FX" and seg.feats[v.idx].sfx_impact), None)
    kin = max((seg.feats[v.idx].kinetic for v in tail), default=0.0)
    vis_val = (1.0 if closeup else 0.0) + 0.6 * kin + (1.8 if sfx_cliff else 0.0) + (0.6 if sfx_imp else 0.0)
    raw += vis_val
    tail_text = " ".join(v.text for v in tail if v.track != "SOUND_FX")
    if RESOLUTION_RX.search(tail_text) and not sfx_cliff:
        raw -= 1.2
        reasons.append("resolution beat in final shot")
    if (last is None or e - last.end_sec > tail_sec) and vis_val < 1.0:
        raw -= 1.0
        reasons.append("limp ending")
    if any(seg.feats[v.idx].transition for v in seg.visuals):
        raw -= 2.5
        reasons.append("contains fade/cut-to-black")
    if ctype in ("visual", "line") and sfx_cliff:
        ctype = "sfx_sting"
    closure, cwhy = closure_signals(seg, tail_sec)
    if closure >= 2:
        raw -= min(4.0, 1.3 * closure)
        reasons.append("scene closes (" + ", ".join(cwhy) + ")")
    vres = visual_resolution(seg, tail_sec)
    if vres:
        raw -= 2.5
        reasons.append("ending resolves (" + "; ".join(vres) + ")")
    resolved = closure >= 2 or bool(vres)
    res_why = (["scene closes: " + ", ".join(cwhy)] if closure >= 2 else []) + vres
    tail_visual = closeup or next((v for v in reversed(tail) if v.track in ("CAMERA", "ACTION")), None)
    return CliffProfile(score=max(0.0, min(10.0, raw)), cliff_type=ctype, line=last, tail_visual=tail_visual,
                        tail_sfx=sfx_cliff or sfx_imp, reasons=reasons, closure=closure, resolved=resolved,
                        resolution_reasons=res_why)


def clarity_score(ctx: BlindContext, leaky_names: Optional[Set[str]] = None) -> Tuple[float, Dict[str, float]]:
    """Standalone cold-viewer clarity. `leaky_names` simulates an evaluator that read the logline."""
    dl = ctx.all_dialogue()
    allfeats: Dict[int, EventFeatures] = {}
    for s in ctx.segments:
        for d in s.dialogue:
            allfeats[id(d)] = s.feats[d.idx]
    anchors: Set[str] = set()
    names: Set[str] = set()
    pronoun_lines = 0
    lore = 0
    conflict = False
    for d in dl:
        f = allfeats[id(d)]
        anchors |= set(f.anchors)
        line_names = spoken_names(d.text)
        names |= line_names
        if PRONOUN_RX.search(d.text) and not f.anchors and not line_names:
            pronoun_lines += 1
        lore += len(LORE_RX.findall(d.text))
        if set(f.acts) & {"accusation", "threat", "command", "reveal"}:
            conflict = True
    line_counts: Dict[str, int] = {}
    for d in dl:
        line_counts[d.speaker] = line_counts.get(d.speaker, 0) + 1
    principals = [k for k, v in line_counts.items() if v >= 2]
    extras = len(line_counts) - len(principals)
    n_sp = len(principals) + (1 if not principals and line_counts else 0)
    visual_conflict = any(s.feats[v.idx].kinetic >= 2.0 for s in ctx.segments for v in s.visuals
                          if v.track in ("ACTION", "SOUND_FX"))
    raw = 2.6
    raw += min(0.85 * len(anchors), 3.4)
    raw += min(0.35 * len(names), 1.05)
    raw += 1.0 if conflict else 0.0
    raw += {0: -1.0, 1: 0.2, 2: 1.0, 3: 1.0, 4: 0.5}.get(n_sp, 0.5 - 0.6 * (n_sp - 4))
    raw -= 0.15 * max(0, extras - 1)
    if visual_conflict:
        raw += 0.7          # a cold viewer *sees* who is attacking whom
    ratio = pronoun_lines / max(len(dl), 1)
    if visual_conflict:
        ratio *= 0.6
    if leaky_names:
        # an evaluator that read the header "knows" who everyone is: pronoun soup and lore look fine
        ratio *= 0.2
        lore = 0
        raw += min(0.5 * len(leaky_names), 1.5)
    raw -= 2.0 * max(0.0, ratio - 0.35)
    raw -= min(0.7 * lore, 2.1)
    headings = []
    for s in ctx.segments:
        for h in s.headings:
            loc = location_key(h)
            if loc not in headings:
                headings.append(loc)
    raw -= 0.45 * max(0, len(headings) - 1)
    if len(ctx.episodes) > 1:
        raw -= 0.4
    for s in ctx.segments[1:]:
        if s.dialogue and s.feats[s.dialogue[0].idx].dependent_opener:
            raw -= 0.6
    diag = {"anchors": len(anchors), "names": len(names), "speakers": n_sp, "extras": extras,
            "visual_conflict": float(visual_conflict), "pronoun_ratio": round(ratio, 2),
            "lore_refs": lore, "conflict": float(conflict), "locations": len(headings)}
    return max(0.0, min(10.0, raw)), diag


def pacing_score(ctx: BlindContext) -> Tuple[float, Dict[str, float]]:
    covered, dead, n_lines = 0.0, 0.0, 0
    for s in ctx.segments:
        dl = s.dialogue
        n_lines += len(dl)
        ivs = sorted((max(d.start_sec, s.start_sec), min(d.end_sec, s.end_sec)) for d in dl)
        busy = list(ivs) + [(max(v.start_sec, s.start_sec), min(v.end_sec, s.end_sec)) for v in s.visuals
                            if s.feats[v.idx].kinetic >= 2.0]
        cur_s = cur_e = None
        for a, b in ivs:
            if cur_e is None or a > cur_e:
                if cur_e is not None:
                    covered += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        if cur_e is not None:
            covered += cur_e - cur_s
        cursor = s.start_sec
        for a, b in sorted(busy):
            if a > cursor:
                dead = max(dead, a - cursor)
            cursor = max(cursor, b)
        dead = max(dead, s.end_sec - cursor)
    dur = max(ctx.total_duration, 1e-6)
    c = covered / dur
    rate = n_lines / dur * 10.0
    raw = 1.5 + 7.0 * c + 0.45 * min(rate, 4.0) - 0.7 * max(0.0, dead - 2.5)
    return max(0.0, min(10.0, raw)), {"dialogue_coverage": round(c, 3), "max_dead_air": round(dead, 2),
                                      "lines_per_10s": round(rate, 2)}


def virality_score(ctx: BlindContext) -> Tuple[float, List[str], Dict[str, int]]:
    hits = trope_hits(ctx.all_text())
    strengths = sorted(((TROPE_WEIGHTS[t] * min(1.0, n / 2.0), t) for t, n in hits.items()), reverse=True)
    vals = [v for v, _ in strengths] + [0.0, 0.0, 0.0]
    max_ar = max((s.feats[d.idx].arousal for s in ctx.segments for d in s.dialogue), default=0.0)
    raw = 1.8 + 3.0 * vals[0] + 2.2 * vals[1] + 1.4 * vals[2] + 0.35 * max_ar
    tags = [t for v, t in strengths if v >= 0.35]
    return max(0.0, min(10.0, raw)), tags, hits


def spoiler_score(episodes: Sequence[int], text: str, title_toks: Set[str]) -> Tuple[float, List[str]]:
    ep = max(episodes) if episodes else 1
    base = 1.0 + 0.15 * (ep - 1) if ep <= 10 else 2.5 + 0.13 * (ep - 10)
    twists = []
    low = text.lower()
    for m in TWIST_RX.finditer(text):
        words = set(re.findall(r"[a-z]+", m.group(0).lower()))
        near = set(re.findall(r"[a-z]+", low[max(0, m.start() - 45): m.end() + 45]))
        if words & title_toks or near & _title_nouns(title_toks):
            continue      # the show title already promises this reveal (e.g. "the rightful heir") -> not a spoiler
        twists.append(m.group(0).lower())
    bump = min(1.2 * len(set(twists)), 2.4) if ep > 10 else 0.0
    return max(0.0, min(10.0, base + bump)), sorted(set(twists))


@dataclass
class BlindScores:
    scores: Dict[str, float]
    hook: HookProfile
    cliff: CliffProfile
    tags: List[str]
    diagnostics: Dict[str, float]


def score_blind(ctx: BlindContext, title_toks: Set[str]) -> BlindScores:
    hook = hook_profile(ctx.segments[0])
    cliff = cliff_profile(ctx.segments[-1])
    clar, cdiag = clarity_score(ctx)
    pace, pdiag = pacing_score(ctx)
    vir, tags, _ = virality_score(ctx)
    spo, twists = spoiler_score(ctx.episodes, ctx.all_text(), title_toks)
    scores = {
        "hook_strength": round(hook.score, 2),
        "cliffhanger_tension": round(cliff.score, 2),
        "standalone_clarity": round(clar, 2),
        "pacing_density": round(pace, 2),
        "trope_virality": round(vir, 2),
        "spoiler_risk": round(spo, 2),
    }
    diag = dict(cdiag)
    diag.update(pdiag)
    diag["twist_terms"] = len(twists)
    diag["closure"] = cliff.closure
    diag["title_promise"] = title_promise(ctx, title_toks)
    return BlindScores(scores=scores, hook=hook, cliff=cliff, tags=tags, diagnostics=diag)


def _title_nouns(title_toks: Set[str]) -> Set[str]:
    return {t for t in title_toks if len(t) > 3 and t not in TITLE_STOP}


TITLE_STOP = {"married", "marry", "with", "that", "this", "from", "into", "after", "before", "when", "were", "broken",
              "secret", "their", "there", "about", "love", "story"}


def title_reveal_nouns(show_title: str) -> Set[str]:
    """The noun the title promises will be revealed: 'My Husband Is a Billionaire' -> {'billionaire'}. Falls back to every title noun when the title has no 'is/was a ...' clause."""
    low = show_title.lower()
    m = re.findall(r"\b(?:is|was|are|were)\s+(?:a|an|the|my|her|his)?\s*([a-z]+(?:\s+[a-z]+)?)", low)
    nouns = {w for grp in m for w in grp.split() if len(w) > 2 and w not in TITLE_STOP}
    return nouns or _title_nouns(title_tokens(show_title))


def identity_reveal_rx(nouns: Set[str]) -> Optional[re.Pattern]:
    """A line that states the title reveal to someone's face: "You're the CEO?", "I am the heir",
    "the rightful heir of ...". Narration about it ("he was revealed to be the heir") does not count."""
    if not nouns:
        return None
    alt = "|".join(sorted(map(re.escape, nouns), key=len, reverse=True))
    return re.compile(r"\b(?:you[\u2019']re|you are|i am|i[\u2019']m|he is|he[\u2019']s|she is|she[\u2019']s|it was|"
                      r"it[\u2019']s|you were)\b[^.?!]{0,25}"
                      r"\b(?:" + alt + r")\b|\b(?:rightful|true|real)\s+(?:" + alt + r")\b", re.I)


def title_promise(ctx: "BlindContext", title_toks: Set[str]) -> int:
    """1 if a spoken line reveals what the show title promises (e.g. the title says 'He Was a Billionaire' and a line
    says 'he is the billionaire'): the title already sells it, so it is a premise reveal, not a spoiler."""
    nouns = _title_nouns(title_toks)
    if not nouns:
        return 0
    rx = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, nouns), key=len, reverse=True)) + r")\b", re.I)
    for d in ctx.all_dialogue():
        if rx.search(d.text) and REVEAL_MARK_RX.search(d.text):
            return 1
    return 0


def composite(scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or config.DEFAULT_WEIGHTS
    raw = sum(w[k] * float(scores.get(k, 0.0)) for k in w)
    return round(max(0.0, min(100.0, raw * 10.0)), 2)
