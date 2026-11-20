"""Stage 1b - blind cold-viewer rating and the copy (deterministic, free).

Guardrail: the rating only ever sees the CAMERA / ACTION / DIALOGUE / SOUND FX blocks inside the promo's own
segment windows. EPISODE LOGLINE and CONTINUITY NOTES are never used; speaker labels are anonymised (viewers don't
see name tags) and character names in visual prose are masked unless the name is spoken inside the clip.

Every candidate gets: six 0-10 scores, the hook and cliffhanger lines (quoted from the clip with their timestamps),
and one stake-driven line on why it works. Stage 2 (src/reranker.py) may re-order the shortlist and rewrite that
line with a model; the timestamps and quotes always come from here.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import config
from .candidate_generator import CandidateGenerator
from .lexicons import (ANCHOR_RX, COMMAND_RX, TROPE_LABELS, TROPE_WEIGHTS, dialogue_acts, dialogue_arousal,
                       emotion_arousal, show_terms_in, trope_hits)
from .schema import Candidate, Promo, ScriptEvent, sec_to_ts
from .show_profile import descriptor_words
from .scoring import (ACT_PRIORITY, HARD_OPEN_ACTS, OPEN_LINE_TAIL_SEC, BlindContext, BlindScores, build_segment,
                      identity_reveal_rx, score_blind)

SMALL_WORDS = {"in", "of", "the", "a", "an", "and", "to", "with", "on", "at"}
HOOK_WHY = {
    "reveal": "a reveal in the opening line",
    "threat": "a threat in the first beat",
    "accusation": "an accusation in the first second",
    "command": "an order in the first second",
    "denial": "a defiant denial in the first second",
    "vow": "a defiant vow in the first beat",
    "plea": "a desperate plea in the first beat",
    "question": "a sharp question in the first beat",
    "verdict": "a verdict in the first second",
    "line": "a charged line in the first beat",
    "kinetic": "physical action inside the first 3 seconds",
    "silent": "a visual-only opening (weak: no line or action in the first 3 s)",
}
CLIFF_Q = {
    "question": None,
    "threat": "will the threat be carried out?",
    "reveal": "what happens now that this is out?",
    "accusation": "how will the accused answer?",
    "verdict": "what happens to the one condemned?",
    "command": "will the order be carried out?",
    "vow": "will the vow be kept?",
    "plea": "will anyone answer the plea?",
    "line": "the next move is withheld",
    "sfx_sting": "the score peaks on an unresolved look",
    "visual": "the shot holds on an unresolved reaction",
}
# one line on why it works, written by code from the detected hook and cut (models wrote generic marketing lines)
HOOK_PHRASE = {"reveal": "{who} drops a reveal in the first {t}", "threat": "{who} threatens in the first {t}",
               "verdict": "{who} passes a verdict in the first {t}",
               "accusation": "{who} accuses in the first {t}", "command": "{who} gives an order in the first {t}",
               "denial": "{who} fights back in the first {t}",
               "vow": "{who} makes a defiant vow in the first {t}", "plea": "{who} begs for help in the first {t}",
               "question": "{who} asks a sharp question in the first {t}",
               "line": "{who} opens on a charged line in the first {t}", "kinetic": "it opens on physical action",
               "silent": "it opens on a quiet shot"}
CLIFF_PHRASE = {"question": "on {who}'s question before anyone answers",
                "verdict": "on {who}'s verdict before anyone reacts",
                "threat": "on {who}'s threat before it is carried out",
                "reveal": "on {who}'s reveal before anyone reacts",
                "accusation": "on {who}'s accusation before the answer",
                "command": "on {who}'s order before it is obeyed", "vow": "on {who}'s vow before it is tested",
                "plea": "on {who}'s plea before anyone answers", "line": "mid-confrontation",
                "sfx_sting": "on a sting over an unresolved look", "visual": "on an unresolved reaction"}
# stock marketing phrases that say nothing about the clip (kept out of every copy line; checked after each run)
STOCK_PHRASES = ["vertical drama fans", "drama fans", "crave", "unmatched", "adrenaline", "short-form", "scroller",
                 "scrolling viewer", "thumb", "irresistible", "fan-favorite", "fan favorite", "highest-converting",
                 "dopamine", "binge", "addictive", "must-watch", "maximum", "maximises", "maximizes", "engagement",
                 "viral", "instantly", "staple", "trope"]
STOCK_RX = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in STOCK_PHRASES) + r")\w*", re.I)
INVENTED_Q_RX = [re.compile(r"[;,:\u2014-]\s*[^;,:\"\u201c\u201d'\u2018\u2019]{3,}\?\s*\.?\s*$"),
                 re.compile(r"(?<=[.!])\s+[A-Z][^.!?\"\u201c\u201d]{3,}\?\s*\.?\s*$")]


def drop_stock_clauses(text: str) -> str:
    """Drop every clause that leans on a stock phrase ('..., stopping scrollers instantly') instead of deleting the
    words and leaving a broken sentence."""
    parts = re.split(r"(?<=[,;:])\s+|\s+(?=\u2014)", text)
    keep = [x for x in parts if not STOCK_RX.search(x)]
    out = " ".join(keep).strip().rstrip(",;:\u2014 ")
    return out


def strip_invented_questions(text: str) -> str:
    """'...lifts the firewood; how will they survive?.' -> '...lifts the firewood' (a question the clip never asks)."""
    out = text.strip()
    for rx in INVENTED_Q_RX:
        out = rx.sub("", out).strip()
    out = re.sub(r"([?!])\.", r"\1", out)
    return re.sub(r"\.{2,}$", ".", out).strip()


def title_case(label: str) -> str:
    words = label.lower().split()
    return " ".join(w if (i and w in SMALL_WORDS) else w[:1].upper() + w[1:] for i, w in enumerate(words))


def display_name(ev: ScriptEvent) -> str:
    raw = title_case(ev.speaker_raw)
    canon = title_case(ev.speaker)
    if canon and canon.lower() not in raw.lower():
        return f"{raw} ({canon})"
    return raw


def short_name(ev: Optional[ScriptEvent]) -> str:
    """'Company Boss (Mr Grey)' -> 'Mr Grey'; 'Lady Maya' -> 'Lady Maya'."""
    if ev is None:
        return "someone"
    canon = title_case(ev.speaker) if ev.speaker else ""
    return canon or title_case(ev.speaker_raw) or "someone"


def short(text: str, n: int = 95) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def quote(text: str, n: int = 110) -> str:
    return '"' + short(text, n) + '"'


ANON_RX = re.compile(r"\[?\b(?:speaker|SPEAKER|Speaker)[ _]?([A-Z])\b\]?('s)?")
LEAD_TS_RX = re.compile(r"^\s*(?:\[?\s*(?:Ep\s?\d+\s*)?(?:S\d+\s*)?(?:cut\s+at\s+)?(?:\+?\d{1,2}:\d{2}(?:\.\d+)?|\+\d+(?:\.\d+)?s)"
                        r"\s*\]?\s*(?:\(\s*\+?\d+(?:\.\d+)?s\s*\))?\s*[:\-\u2013\u2014,]?\s*)+", re.I)
SEG_REF_RX = re.compile(r"\bS(\d)\b")


def clean_model_copy(text: str, names: Dict[str, str]) -> str:
    """Model copy -> marketer copy: drop the leading timestamp/segment chatter (the UI shows exact timestamps),
    and replace anonymised speaker tags with the character's name."""
    text = LEAD_TS_RX.sub("", str(text or "")).strip()
    text = ANON_RX.sub(lambda m: names.get(m.group(1), "someone") + (m.group(2) or ""), text)
    text = text.replace("[someone]", "someone")
    text = SEG_REF_RX.sub(lambda m: f"segment {m.group(1)}", text)
    return (text[:1].upper() + text[1:]) if text else text


def copy_line(ts: str, said: Optional[str], desc: str) -> str:
    """Standard copy format: [timestamp] "line as spoken" - description."""
    desc = desc.strip().rstrip(".")
    end = "" if desc.endswith(("?", "!")) else "."
    return f'[{ts}] {said} \u2014 {desc}{end}' if said else f"[{ts}] {desc}{end}"


VERDICT_RX = re.compile(r"^\s*[\"\u201c']?\s*you(?:'re| are)\s+(?:\w+\s+)?(?:disowned|fired|banished|finished|"
                        r"expelled|dismissed|under arrest|not welcome|dead to (?:me|us)|nothing|no (?:daughter|son|"
                        r"wife|husband) of)", re.I)
DENOUNCE_RX = re.compile(r"\b(?:this|that|he|she)\s+(?:man|woman|one)?\s*(?:is|was)\s+(?:an?|the)\s+(?:imposter|impostor|"
                         r"liar|fraud|thief|murderer|traitor|cheat)\b", re.I)
QUESTION_FIRST_RX = re.compile(r"^\s*[\"\u201c']?\s*(?:who|what|where|when|why|how|which|is|are|do|does|did|can|"
                               r"could|will|would|have|has)\b[^!]*\?\s*[\"\u201d']?\s*$", re.I)


def line_type(text: str) -> str:
    """How the copy describes one spoken line: the act a viewer hears first ("You framed me!" is an accusation,
    "Get out of my house!" an order, "You are fired!" a verdict, "Who is the father?" a question)."""
    acts = dialogue_acts(text)
    if re.match(r"\s*[\"\u201c']?\s*(?:i did not|i didn't|i never|i am not a|i'm not a)\b", text, re.I) and "!" in text:
        return "denial"
    if VERDICT_RX.match(text):
        return "verdict"
    if QUESTION_FIRST_RX.match(text):
        return "question"
    if DENOUNCE_RX.search(text) or ("accusation" in acts and re.match(r"\s*[\"\u201c']?\s*you\b", text, re.I)):
        return "accusation"
    if "command" in acts and COMMAND_RX.match(text):
        return "command"
    return next((a for a in ACT_PRIORITY if a in acts), "line")


def hook_line(hp: Any, seg0: Any) -> Optional[ScriptEvent]:
    """The spoken line the copy quotes as the hook: the strongest line in the first 3 s (not a '...' fragment),
    else the first line. Hook copy and why-it-works both use it, so they always describe the same line."""
    opening = [d for d in seg0.dialogue if d.start_sec <= seg0.start_sec + config.HOOK_WINDOW_SEC + 1e-6]
    ev = hp.event
    if (ev is not None and ev.track == "DIALOGUE" and ev.start_sec <= seg0.start_sec + config.HOOK_WINDOW_SEC + 1e-6
            and not re.match(r"\s*(?:\.\.\.|\u2026)", ev.text)):
        return ev
    return opening[0] if opening else None



# ---- stake-driven "why it works" (code): situation + who acts in the first second + the stake left hanging --------
HOOK_VERB = {"accusation": "lashes out", "command": "gives the order", "plea": "begs for help", "threat": "threatens",
             "reveal": "drops a secret", "question": "demands answers", "denial": "fights back",
             "verdict": "passes a verdict", "vow": "makes a vow", "line": "opens the confrontation"}
CUT_AFTER = {"question": "before anyone answers", "threat": "with the threat still hanging", "reveal": "before anyone reacts",
             "accusation": "before the answer", "command": "before it is obeyed", "plea": "before anyone answers",
             "verdict": "before anyone reacts", "vow": "before it is tested", "denial": "before anyone answers",
             "line": "with nothing settled"}


VOCATIVE_RX = re.compile(r"^(?:(?:oh|ah|please|no|yes)[,!]?\s+)?(?:[A-Z][\w'\u2019-]*\s*){1,3}[!?.,]*$")


def stake_fragment(text: str, n_words: int = 9) -> str:
    """The sentence of a line that carries the stake. A short line is quoted whole; otherwise the sentence with a
    title's twist wins first, then a family / crime / money word or one of this show's own high-stakes words, then a
    threat, plea, reveal, question or the most heated sentence; a bare name ("Lady Maya!") never does. Trimmed to
    its last n words."""
    line = text.strip(" \"\u201c\u201d'")
    if len(line.split()) <= n_words:
        return line
    sents = [s.strip(" \"\u201c\u201d'") for s in re.split(r"(?<=[.;!?\u2014\u2026])\s+", line) if s.strip(" \"'")]
    if not sents:
        return ""

    def weight(s: str) -> float:
        if VOCATIVE_RX.match(s):
            return -3.0
        w = 3.0 * bool(show_terms_in(s, "premise"))           # the title's twist, said out loud
        w += 2.0 * bool(ANCHOR_RX.search(s) or show_terms_in(s, "anchors"))
        w += 1.0 * bool(set(dialogue_acts(s)) & {"threat", "reveal", "plea", "question", "accusation"})
        return w + 0.5 * (len(s.split()) >= 3) + 0.4 * dialogue_arousal(s, [])
    best = max(range(len(sents)), key=lambda k: (weight(sents[k]), k))
    frag = sents[best].rstrip(",;\u2014 ")
    words = frag.split()
    return " ".join(words[-n_words:]) if len(words) > n_words else frag


class PromoEvaluator:
    def __init__(self, gen: CandidateGenerator, show_title: str, brief: Any = None):
        self.gen = gen
        self.title_toks = gen.title_toks
        self.show_title = show_title
        self.brief = brief
        self.descriptors = descriptor_words(gen.episodes)     # role words + descriptions learned from this show
        self.cast_names = self._cast_names()
        self.premise_rx = identity_reveal_rx(self.gen.reveal_nouns)

    # ------------------------------------------------------------------ name masking
    def _cast_names(self) -> Dict[str, str]:
        """Character proper names (from speaker labels confirmed by the headers) -> canonical entity.
        Used only to MASK names in visual prose; header text itself is never shown to the evaluator."""
        names: Dict[str, str] = {}
        labels = {e.speaker for idx in self.gen.index.values() for e in idx.dialogue}
        labels |= {e.speaker_raw for idx in self.gen.index.values() for e in idx.dialogue}
        header_text = " ".join(idx.ep.logline + " " + idx.ep.continuity_notes for idx in self.gen.index.values())
        header_names = set(re.findall(r"\b([A-Z][a-z]{2,})\b", header_text))
        for lab in labels:
            for tok in re.findall(r"[A-Z][A-Z']+", lab):
                t = tok.replace("'S", "").title()
                if t in header_names and t not in self.descriptors:
                    names.setdefault(t, lab if len(lab.split()) == 1 else
                                     next((c for c in labels if c == tok.replace("'S", "")), lab))
        return names

    def _mask(self, text: str, letters: Dict[str, str], spoken: Set[str]) -> str:
        def rep(m: re.Match) -> str:
            name = m.group(1)
            if name in spoken:
                return m.group(0)
            ent = self.cast_names.get(name)
            tag = f"[Speaker {letters[ent]}]" if ent in letters else "[someone]"
            return tag + (m.group(2) or "")
        if not self.cast_names:
            return text
        rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, self.cast_names), key=len, reverse=True)) + r")('s)?\b")
        return rx.sub(rep, text)

    # ------------------------------------------------------------------ blind context
    def blind_context(self, cand: Candidate) -> BlindContext:
        return BlindContext([build_segment(self.gen.index[s.episode], s.start_sec, s.end_sec) for s in cand.segments])

    @staticmethod
    def speaker_letters(ctx: BlindContext) -> Dict[str, str]:
        letters: Dict[str, str] = {}
        for seg in ctx.segments:
            for d in seg.dialogue:
                if d.speaker not in letters:
                    letters[d.speaker] = chr(ord("A") + len(letters))
        return letters

    def letter_names(self, ctx: BlindContext) -> Dict[str, str]:
        """'A' -> 'Maya' (the name a marketer should read in the copy)."""
        return {v: title_case(k) for k, v in self.speaker_letters(ctx).items()}

    def render_prompt(self, cand: Candidate, ctx: Optional[BlindContext] = None, for_batch: bool = False) -> str:
        ctx = ctx or self.blind_context(cand)
        letters = self.speaker_letters(ctx)
        spoken: Set[str] = set()
        for seg in ctx.segments:
            for d in seg.dialogue:
                spoken |= set(re.findall(r"\b([A-Z][a-z]{2,})\b", d.text))
        lines = [f"PROMO CLIP - type={cand.promo_type}, {len(ctx.segments)} segment(s), "
                 f"total {ctx.total_duration:.2f}s (wall-clock). Segments play back-to-back in this order."]
        offset = 0.0
        for n, seg in enumerate(ctx.segments, 1):
            lines.append(f"\nSEGMENT {n}/{len(ctx.segments)} - {seg.duration:.2f}s - setting: "
                         f"{' / '.join(seg.headings) or 'unknown'}")
            dl_ids = {id(d) for d in seg.dialogue}
            for e in seg.events:
                if e.track == "DIALOGUE" and id(e) not in dl_ids:
                    continue
                rel = max(0.0, e.start_sec - seg.start_sec) + offset
                clock = f"S{n} {e.start_ts if e.start_sec >= seg.start_sec else sec_to_ts(seg.start_sec)}"
                if e.track == "DIALOGUE":
                    tags = ", ".join(e.emotion_tags)
                    who = f"SPEAKER {letters[e.speaker]}" + (" (off-screen)" if e.qualifiers else "")
                    lines.append(f"[{clock} | +{rel:05.2f}s] {who} ({tags}): \"{e.text}\"")
                else:
                    vis = short(self._mask(e.text, letters, spoken), 110)
                    lines.append(f"[{clock} | +{rel:05.2f}s] {e.track.replace('_', ' ')}: {vis}")
            lines.append(f"[S{n} cut at {sec_to_ts(seg.end_sec)}]")
            offset += seg.duration
        if not for_batch:
            lines.append("\nReturn the JSON object only.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ copy
    def compose_why(self, cand: Candidate, bs: BlindScores, ctx: BlindContext, premise: bool) -> str:
        """One line on why it works: the situation, who acts in the first second, and the stake the cut leaves
        hanging, quoted from the clip. No templates about rules, no invented questions."""
        hp, cp = bs.hook, bs.cliff
        seg0, last = ctx.segments[0], cand.segments[-1]
        hook_ev = hook_line(hp, seg0) if hp.hook_type != "kinetic" else None
        if hook_ev is not None:
            htype = line_type(hook_ev.text)
            opener = f"{short_name(hook_ev)} {HOOK_VERB.get(htype, HOOK_VERB['line'])} in the first second"
        elif hp.hook_type == "kinetic":
            opener = "it opens on physical action"
        else:
            opener = "it opens on a quiet shot"
        cline = cp.line if (cp.line is not None and last.end_sec - cp.line.end_sec <= config.CLIFF_WINDOW_SEC) else None
        if cline is not None:
            frag = stake_fragment(cline.text)
            ctype = line_type(frag) if frag else line_type(cline.text)     # the quoted words decide the phrasing
            who = "" if (hook_ev is not None and cline.speaker == hook_ev.speaker) else f"{short_name(cline)}'s "
            cut = f"it cuts on {who}\u201c{frag}\u201d {CUT_AFTER.get(ctype, CUT_AFTER['line'])}"
        else:
            cut = "it cuts on an unresolved reaction"
        if premise and self.brief is not None and getattr(self.brief, "premise_twist", ""):
            label = f"Sells the premise ({self.brief.premise_twist})"
        else:          # name the story type only when the hook and cut lines clearly show it (2+ keyword hits)
            beat_text = " ".join(x.text for x in (hook_ev, cline) if x is not None)
            beat_hits = {k: v for k, v in trope_hits(beat_text).items() if v >= 2}
            label = TROPE_LABELS[max(beat_hits, key=lambda k: (beat_hits[k] * TROPE_WEIGHTS[k], k))] if beat_hits else ""
        text = f"{opener}, and {cut}."
        text = text[:1].upper() + text[1:]
        return f"{label}: {text}" if label else text

    def heuristic_copy(self, cand: Candidate, bs: BlindScores, ctx: BlindContext
                       ) -> Tuple[str, str, str, str, str]:
        hp, cp = bs.hook, bs.cliff
        first, last = cand.segments[0], cand.segments[-1]
        ep_tag = (lambda s: f"Ep{s.episode} ") if len(cand.episodes) > 1 else (lambda s: "")
        seg0 = ctx.segments[0]
        opening = [d for d in seg0.dialogue if d.start_sec <= seg0.start_sec + config.HOOK_WINDOW_SEC + 1e-6][:2]
        # hook: quote the actual opening line (what the viewer hears in the first 3 s)
        if hp.hook_type != "kinetic" and opening:
            # quote the line that carries the hook and describe that same line (not the first line with the best
            # line's label)
            ev = hook_line(hp, seg0) or opening[0]
            own = line_type(ev.text)
            hook_ts = ev.start_ts
            tags = f" ({', '.join(t.lower() for t in ev.emotion_tags)})" if ev.emotion_tags else ""
            hook = copy_line(f"{ep_tag(first)}{ev.start_ts}", quote(ev.text, 110),
                             f"{display_name(ev)}{tags}: {HOOK_WHY.get(own, HOOK_WHY['line'])}")
        elif hp.event is not None:
            ev = hp.event
            hook_ts = ev.start_ts if ev.start_sec >= first.start_sec else first.start_timestamp
            follow = f", then {display_name(opening[0])}: {quote(opening[0].text, 70)}" if opening else ""
            hook = copy_line(f"{ep_tag(first)}{hook_ts}", None, f"{short(ev.text, 120).rstrip('.')}{follow} - "
                                                                 f"{HOOK_WHY['kinetic']}")
        else:
            hook_ts = first.start_timestamp
            hook = copy_line(f"{ep_tag(first)}{first.start_timestamp}", None, HOOK_WHY["silent"])
        # cliffhanger
        bits = []
        if cp.line is not None and last.end_sec - cp.line.end_sec <= config.CLIFF_WINDOW_SEC:
            bits.append(f"{display_name(cp.line)}: {quote(cp.line.text, 120)}")
        if cp.tail_visual is not None:
            bits.append(f"held on {lower_first(short(cp.tail_visual.text, 100).rstrip('.'))}")
        if cp.tail_sfx is not None:
            bits.append(f"under {lower_first(short(cp.tail_sfx.text, 70).rstrip('.'))}")
        q = CLIFF_Q.get(cp.cliff_type)
        if q is None:
            q = f"the question {quote(cp.line.text, 80)} is left hanging" if cp.line is not None else CLIFF_Q["line"]
        cliff_ts = last.end_timestamp
        said = quote(cp.line.text, 120) if (cp.line is not None and last.end_sec - cp.line.end_sec <= config.CLIFF_WINDOW_SEC) else None
        rest = [b for b in bits if not (said and b.startswith(display_name(cp.line)))]
        who = f"{display_name(cp.line)}; " if said else ""
        cliff = copy_line(f"cut {ep_tag(last)}{last.end_timestamp}", said,
                          who + "; ".join(rest or ["held on the reaction"]) + f" - unresolved: {q}")
        why = ""          # composed in _to_promo, once we know whether the clip is a title reveal
        return hook, cliff, why, hook_ts, cliff_ts

    # ------------------------------------------------------------------ signals used by the ranker
    def premise_reveal(self, cand: Candidate, ctx: BlindContext, bs: BlindScores) -> bool:
        """The clip sells the title's twist: its last segment states it to a face ("You're the CEO?", "the rightful
        heir") within 6 s of the cut, and no other late twist is in it. The title already sells this, so it is a hook,
        not a spoiler (assumption in the doc)."""
        if self.premise_rx is None or int(bs.diagnostics.get("twist_terms", 0)):
            return False
        seg = ctx.segments[-1]
        return any(self.premise_rx.search(d.text) and seg.end_sec - d.end_sec <= 6.0 for d in seg.dialogue)

    @staticmethod
    def hard_open_end(ctx: BlindContext) -> bool:
        """The clip ends on a live line: a threat, question, plea or reveal in its last 1.5 s."""
        seg = ctx.segments[-1]
        if not seg.dialogue:
            return False
        last = seg.dialogue[-1]
        f = seg.feats[last.idx]
        return (seg.end_sec - last.end_sec <= OPEN_LINE_TAIL_SEC + 1e-6 and emotion_arousal(last.emotion_tags) >= 0
                and bool(set(f.acts) & HARD_OPEN_ACTS or last.text.rstrip().endswith("?")))

    def sequences_of(self, cand: Candidate) -> List[str]:
        out: List[str] = []
        for sg in cand.segments:
            idx = self.gen.index[sg.episode]
            ev = next((e for e in idx.events if e.start_sec >= sg.start_sec - 1e-6), None)
            sid = self.gen.sequences.get((sg.episode, ev.scene_id)) if ev is not None else None
            if sid and sid not in out:
                out.append(sid)
        return out

    # ------------------------------------------------------------------ main
    def _to_promo(self, cand: Candidate, bs: BlindScores, ctx: BlindContext) -> Promo:
        hook, cliff, _, hook_ts, cliff_ts = self.heuristic_copy(cand, bs, ctx)
        premise = self.premise_reveal(cand, ctx, bs)
        scores = dict(bs.scores)
        signals = dict(cand.signals)
        if premise:          # the title supplies the context a stranger would otherwise miss
            bonus = round(min(10.0, scores["standalone_clarity"] + config.PREMISE_CLARITY_BONUS)
                          - scores["standalone_clarity"], 2)
            scores["standalone_clarity"] = round(scores["standalone_clarity"] + bonus, 2)
            signals["title_context_bonus"] = bonus
        why = self.compose_why(cand, bs, ctx, premise)
        signals["hook_pass"] = bool(bs.hook.passes_gate)
        signals["hook_gate_reasons"] = list(bs.hook.reasons)
        signals["resolved"] = bool(bs.cliff.resolved)
        signals["resolution_reasons"] = list(bs.cliff.resolution_reasons)
        signals["hard_open_end"] = self.hard_open_end(ctx)
        signals["premise_reveal"] = premise
        signals["sequences"] = self.sequences_of(cand)
        seg0 = ctx.segments[0]
        signals["hook_window_lines"] = [d.text for d in seg0.dialogue
                                        if d.start_sec <= seg0.start_sec + config.HOOK_WINDOW_SEC + 1e-6]
        signals["clip_lines"] = [d.text for s in ctx.segments for d in s.dialogue]
        signals["hook_type"] = bs.hook.hook_type
        signals["cliff_type"] = bs.cliff.cliff_type
        signals["hook_reasons"] = bs.hook.reasons
        signals["cliff_reasons"] = bs.cliff.reasons
        signals["selection_score"] = cand.selection_score
        signals["closure"] = bs.cliff.closure
        signals["closure_reasons"] = [r for r in bs.cliff.reasons if r.startswith("scene closes")]
        signals["title_promise"] = int(bs.diagnostics.get("title_promise", 0))
        signals["twist_terms"] = int(bs.diagnostics.get("twist_terms", 0))
        signals["speaker_names"] = self.letter_names(ctx)
        return Promo(promo_id=cand.candidate_id, promo_type=cand.promo_type,
                     segments=[s.to_dict() for s in cand.segments], total_duration_sec=cand.total_duration_sec,
                     hook=hook, cliffhanger=cliff, why_it_works=why, scores=scores,
                     hook_timestamp=hook_ts, cliffhanger_timestamp=cliff_ts, episodes=cand.episodes,
                     arc=cand.arc, arc_tags=list(bs.tags), scored_by="heuristic:heuristic-v1", signals=signals)

    def evaluate(self, cands: Sequence[Candidate], progress: Optional[Callable[[str], None]] = None) -> List[Promo]:
        """Rate every candidate blind and write its copy (code only; stage 2 may re-order the shortlist)."""
        self.ctxs: Dict[str, BlindContext] = {}
        self.cands: Dict[str, Candidate] = {c.candidate_id: c for c in cands}
        out = []
        for c in cands:
            ctx = self.blind_context(c)
            self.ctxs[c.candidate_id] = ctx
            out.append(self._to_promo(c, score_blind(ctx, self.title_toks), ctx))
        return out
