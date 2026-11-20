"""Stage 2 - candidate windows + multi-segment stitching (deterministic heuristics + signal scoring).

Three promo topologies, all snapped to real block boundaries (so every timestamp exists verbatim
in the source .txt) and all with wall-clock totals in [20, 60] s:

  A. continuous            one stretch; 3-second hook gate; cliffhanger snap; NMS de-duplication
  B. multi_segment_intra   hook/setup segment + climax segment from the same episode, skipping the middle
  C. multi_segment_cross   early inciting wound + later escalation/payback tease sharing core characters
  P. premise montage       a hook from the show's setup + the moment the title's twist is said on screen
                           (never from the finale), so at least one promo sells what the title promises

tighten() then moves every segment start to 0.3 s before its first spoken line or physical action, so no promo
opens on dead air. The new start is recorded with the verbatim block it leans on (anchor_timestamp).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import config
from .lexicons import KINETIC_ACTION_RX, PAYOFF_TAGS, SETUP_TAGS, TROPE_WEIGHTS, trope_hits
from .parser import EpisodeIndex
from .schema import Candidate, Episode, Segment, sec_to_ts
from .show_profile import ShowProfile, build_profile, story_sequences
from .scoring import (BlindContext, build_segment, cliff_profile, composite, hook_profile,
                      identity_reveal_rx, pacing_score, score_blind, title_reveal_nouns, title_tokens)

HOOK_MIN = 5.2          # minimum hook score for a promo opening (gate also requires a line/kinetic beat)
REHOOK_MIN = 4.2        # minimum hook score for the first beat of a jump-cut segment


@dataclass
class GenStats:
    windows_evaluated: int = 0
    starts_total: int = 0
    starts_passing_gate: int = 0
    ends_total: int = 0
    continuous_kept: int = 0
    intra_kept: int = 0
    cross_kept: int = 0
    per_episode_windows: Dict[int, int] = field(default_factory=dict)


def _iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _span_iou(c1: Candidate, c2: Candidate) -> float:
    """Temporal IoU over the union of segments in the same episode(s)."""
    best = 0.0
    for s1 in c1.segments:
        for s2 in c2.segments:
            if s1.episode == s2.episode:
                best = max(best, _iou((s1.start_sec, s1.end_sec), (s2.start_sec, s2.end_sec)))
    return best


def nms(cands: List[Candidate], iou_thr: float, keep: int) -> List[Candidate]:
    out: List[Candidate] = []
    for c in sorted(cands, key=lambda x: -x.selection_score):
        if all(_span_iou(c, o) < iou_thr for o in out):
            out.append(c)
        if len(out) >= keep:
            break
    return out


def finale_first_episode(episodes: Sequence[int], share: float = config.FINALE_SHARE) -> int:
    """First episode of the finale (the last 10%). Shows under 5 episodes have no finale rule."""
    eps = sorted(episodes)
    if not eps:
        return 1
    n = eps[-1] - eps[0] + 1
    if n < 5:
        return eps[-1] + 1
    return eps[-1] - math.ceil(share * n) + 1


def first_quarter_last(episodes: Sequence[int], share: float = 0.25) -> int:
    eps = sorted(episodes)
    return (eps[0] - 1 + max(1, math.ceil(share * (eps[-1] - eps[0] + 1)))) if eps else 1


class CandidateGenerator:
    def __init__(self, episodes: Sequence[Episode], show_title: str, min_dur: float = config.MIN_DURATION,
                 max_dur: float = config.MAX_DURATION, weights: Optional[Dict[str, float]] = None,
                 profile: Optional[ShowProfile] = None, brief=None):
        self.episodes = list(episodes)
        self.profile = profile or build_profile(self.episodes, show_title)
        self.index: Dict[int, EpisodeIndex] = {ep.episode_number: EpisodeIndex(ep) for ep in self.episodes}
        self.title_toks = title_tokens(show_title, brief.twist_terms if brief is not None else ())
        self.brief = brief
        # the words that state the title's twist on screen: from the show brief, else from the title itself
        self.reveal_nouns = set(getattr(brief, "twist_terms", []) or []) or title_reveal_nouns(show_title)
        self.setup_terms = set(getattr(brief, "setup_terms", []) or [])
        self.finale_from = finale_first_episode(sorted(self.index))
        self.sequences = story_sequences(self.episodes)
        self.min_dur, self.max_dur = min_dur, max_dur
        self.weights = weights or config.DEFAULT_WEIGHTS
        self.stats = GenStats()
        self._hook_cache: Dict[Tuple[int, float], float] = {}
        self._cliff_cache: Dict[Tuple[int, float], float] = {}

    # ---------------------------------------------------------------- per-point signals
    def hook_at(self, ep: int, t: float) -> Tuple[float, bool]:
        idx = self.index[ep]
        key = (ep, t)
        if key not in self._hook_cache:
            seg = build_segment(idx, t, min(t + 20.0, idx.ep.duration_sec))
            hp = hook_profile(seg)
            self._hook_cache[key] = hp.score if hp.passes_gate else -hp.score
        v = self._hook_cache[key]
        return abs(v), v >= 0 and abs(v) > 0

    def cliff_at(self, ep: int, t: float) -> float:
        key = (ep, t)
        if key not in self._cliff_cache:
            idx = self.index[ep]
            seg = build_segment(idx, max(0.0, t - 12.0), t)
            self._cliff_cache[key] = cliff_profile(seg).score
        return self._cliff_cache[key]

    def stage2_bonus(self, idx: EpisodeIndex, end: float) -> Tuple[float, List[str]]:
        """Code-only signals an editor knows but a cold viewer can't: withheld answer / episode-end sting."""
        notes, bonus = [], 0.0
        last = idx.last_dialogue_before(end)
        nxt = idx.next_dialogue_after(end - 1e-6)
        if last is not None and nxt is not None and nxt.start_sec - end <= 3.0 and "question" in idx.feat[last.idx].acts:
            bonus += 0.6
            notes.append("answer withheld by the cut")
        if abs(idx.transition_start - end) <= 1.6:
            bonus += 0.5
            notes.append("lands on the episode's scripted cliffhanger")
        return bonus, notes

    def _ctx(self, spans: Iterable[Tuple[int, float, float]]) -> BlindContext:
        return BlindContext([build_segment(self.index[ep], s, e) for ep, s, e in spans])

    def _duration_pref(self, total: float) -> float:
        lo, hi = config.SWEET_SPOT
        if total < lo:
            return -0.15 * (lo - total)
        if total > hi:
            return -0.12 * (total - hi)
        return 0.0

    def _seg(self, idx: EpisodeIndex, s: float, e: float, role: str) -> Segment:
        s_ts = dict(idx.clean_starts).get(s) or next(ev.start_ts for ev in idx.events if abs(ev.start_sec - s) < 1e-6)
        e_ts = dict(idx.clean_ends).get(e) or next(ev.end_ts for ev in idx.events if abs(ev.end_sec - e) < 1e-6)
        return Segment(episode=idx.ep.episode_number, start_timestamp=s_ts, end_timestamp=e_ts,
                       start_sec=s, end_sec=e, role=role)

    def _good_starts(self, ep: int, k: int, min_hook: float = HOOK_MIN) -> List[Tuple[float, float]]:
        idx = self.index[ep]
        out = []
        for t, _ in idx.clean_starts:
            h, ok = self.hook_at(ep, t)
            if ok and h >= min_hook:
                out.append((h, t))
        out.sort(reverse=True)
        return [(t, h) for h, t in out[:k]]

    def _good_ends(self, ep: int, k: int) -> List[Tuple[float, float]]:
        idx = self.index[ep]
        out = sorted(((self.cliff_at(ep, t), t) for t, _ in idx.clean_ends if t > 5.0), reverse=True)
        return [(t, c) for c, t in out[:k]]

    # ---------------------------------------------------------------- Type A
    def continuous(self, ep: int) -> List[Candidate]:
        idx = self.index[ep]
        self.stats.starts_total += len(idx.clean_starts)
        self.stats.ends_total += len(idx.clean_ends)
        gate_ok = [t for t, _ in idx.clean_starts if self.hook_at(ep, t)[1]]
        self.stats.starts_passing_gate += len(gate_ok)
        cands: List[Candidate] = []
        n = 0
        # relax the hook threshold only if a quiet episode would otherwise yield nothing
        for min_hook, n_ends in ((HOOK_MIN, 18), (3.0, 30), (0.0, 60)):
            starts = self._good_starts(ep, 16, min_hook)
            ends = self._good_ends(ep, n_ends)
            if any(self.min_dur <= e - s <= self.max_dur for s, _ in starts for e, _ in ends):
                break
        for s, _h in starts:
            for e, _c in ends:
                dur = e - s
                if not (self.min_dur - 1e-6 <= dur <= self.max_dur + 1e-6):
                    continue
                n += 1
                ctx = self._ctx([(ep, s, e)])
                bs = score_blind(ctx, self.title_toks)
                bonus, notes = self.stage2_bonus(idx, e)
                sel = composite(bs.scores, self.weights) + 10 * 0.1 * bonus + self._duration_pref(dur)
                cands.append(Candidate(candidate_id="", promo_type="continuous",
                                       segments=[self._seg(idx, s, e, "main")],
                                       signals={"stage2_notes": notes, "blind": bs.scores, "tags": bs.tags,
                                                "diagnostics": bs.diagnostics},
                                       arc_tags=bs.tags, selection_score=round(sel, 3)))
        self.stats.windows_evaluated += n
        self.stats.per_episode_windows[ep] = self.stats.per_episode_windows.get(ep, 0) + n
        kept = nms(cands, config.NMS_IOU, config.MAX_CONTINUOUS_PER_EP)
        for i, c in enumerate(kept, 1):
            c.candidate_id = f"PROMO_EP{ep:02d}_CONT_{i:02d}"
        self.stats.continuous_kept += len(kept)
        return kept

    # ---------------------------------------------------------------- title reveals, cut at the peak
    def premise_cuts(self, ep: int) -> List[Candidate]:
        """The moment the title's promise is said to a character's face ("You're the CEO?"), cut no more than
        4.5 s after the reveal line, so the promo sells the premise instead of running on into what follows."""
        rx = identity_reveal_rx(self.reveal_nouns)
        if rx is None:
            return []
        idx = self.index[ep]
        out: List[Candidate] = []
        ends_all = [t for t, _ in idx.clean_ends]
        for d in idx.dialogue:
            if not rx.search(d.text):
                continue
            ends = [t for t in ends_all if d.end_sec - 1e-6 <= t <= d.end_sec + 4.5]
            best = None
            for e in ends:
                for s, h in self._good_starts(ep, 10_000, 0.0):
                    dur = e - s
                    if not (self.min_dur - 1e-6 <= dur <= min(40.0, self.max_dur) + 1e-6):
                        continue
                    ctx = self._ctx([(ep, s, e)])
                    bs = score_blind(ctx, self.title_toks)
                    sel = composite(bs.scores, self.weights) + self._duration_pref(dur)
                    if best is None or sel > best[0]:
                        best = (sel, s, e, bs)
            if best is None:
                continue
            sel, s, e, bs = best
            if any(abs(c.segments[0].start_sec - s) < 1e-6 and abs(c.segments[0].end_sec - e) < 1e-6 for c in out):
                continue
            out.append(Candidate(candidate_id=f"PROMO_EP{ep:02d}_REVEAL_{len(out) + 1:02d}", promo_type="continuous",
                                 segments=[self._seg(idx, s, e, "main")],
                                 signals={"stage2_notes": [f"title reveal cut at its peak: \u201c{d.text[:60]}\u201d"],
                                          "blind": bs.scores, "tags": bs.tags, "diagnostics": bs.diagnostics,
                                          "premise_cut": True},
                                 arc_tags=bs.tags, selection_score=round(sel, 3)))
        self.stats.windows_evaluated += len(out)
        return out

    # ---------------------------------------------------------------- Type P: premise montage
    def premise_montages(self, keep: int = 3) -> List[Candidate]:
        """Setup + twist in one promo: a strong-hook beat from the first quarter (preferring one that states the
        setup, e.g. the marriage) followed by the moment the title's twist is said to a face, cut within 4.5 s of
        that line. Twist beats come from before the finale. The same-villain guard applies, as for any joined cut."""
        rx = identity_reveal_rx(self.reveal_nouns)
        eps = sorted(self.index)
        if rx is None or len(eps) < 2:
            return []
        fq = first_quarter_last(eps, config.POSTING_RULES["first_quarter"])
        twists: List[Tuple[float, int, float, float, str]] = []
        for ep in eps:
            if ep >= self.finale_from:
                continue
            idx = self.index[ep]
            starts = [t for t, _ in idx.clean_starts]
            ends_all = [t for t, _ in idx.clean_ends]
            for d in idx.dialogue:
                if not rx.search(d.text):
                    continue
                for e in (t for t in ends_all if d.end_sec - 1e-6 <= t <= d.end_sec + 4.5):
                    for s in starts:
                        if 8.0 <= e - s <= 26.0 and s <= d.start_sec - 2.0:
                            h, ok = self.hook_at(ep, s)
                            twists.append(((h if ok else 0.0) + self.cliff_at(ep, e), ep, s, e, d.text))
        twists = sorted(twists, reverse=True)
        best_twists: List[Tuple[float, int, float, float, str]] = []
        for tw in twists:
            if all(not (tw[1] == b[1] and abs(tw[3] - b[3]) < 5.0) for b in best_twists):
                best_twists.append(tw)
            if len(best_twists) >= 4:
                break
        setups: List[Tuple[float, int, float, float]] = []
        for ep in eps:
            if ep > fq:
                break
            idx = self.index[ep]
            for s, h in self._good_starts(ep, 12):
                for e, _ in idx.clean_ends:
                    if 8.0 <= e - s <= 22.0:
                        text = " ".join(x.text for x in idx.events_in(s, e)).lower()
                        says_setup = any(re.search(r"\b" + re.escape(w), text) for w in self.setup_terms)
                        setups.append((h + 0.4 * self.cliff_at(ep, e) + (1.0 if says_setup else 0.0), ep, s, e))
        setups = sorted(setups, reverse=True)[:8]
        out: List[Candidate] = []
        for _, tep, s2, e2, line in best_twists:
            for _, sep, s1, e1 in setups:
                if (sep, e1) >= (tep, s2) or (sep == tep and s2 - e1 < 3.0):
                    continue
                total = (e1 - s1) + (e2 - s2)
                if not (max(25.0, self.min_dur) <= total <= min(55.0, self.max_dur)):
                    continue
                if self._humbled_antagonists(tep, s2, e2) - self._entities(sep, s1, e1):
                    continue          # same rule as every joined cut: no payback from a villain the setup never showed
                ctx = self._ctx([(sep, s1, e1), (tep, s2, e2)])
                bs = score_blind(ctx, self.title_toks)
                sel = composite(bs.scores, self.weights) + self._duration_pref(total)
                i1, i2 = self.index[sep], self.index[tep]
                out.append(Candidate(
                    candidate_id="", promo_type="multi_segment_intra" if sep == tep else "multi_segment_cross",
                    segments=[self._seg(i1, s1, e1, "premise_setup"), self._seg(i2, s2, e2, "premise_twist")],
                    signals={"stage2_notes": [f"premise montage: setup Ep{sep} + twist Ep{tep} \u201c{line[:60]}\u201d"],
                             "blind": bs.scores, "tags": bs.tags, "diagnostics": bs.diagnostics,
                             "premise_montage": True},
                    arc="premise", arc_tags=bs.tags, selection_score=round(sel, 3)))
        self.stats.windows_evaluated += len(out)
        kept = nms(out, 0.5, keep)
        for i, c in enumerate(kept, 1):
            c.candidate_id = f"PROMO_EP{c.segments[-1].episode:02d}_PREMISE_{i:02d}"
        return kept

    # ---------------------------------------------------------------- tight starts
    def tighten(self, cands: Sequence[Candidate], pre_roll: float = config.PRE_ROLL_SEC) -> int:
        """Start every segment pre_roll seconds before its first spoken line or physical action (looked for inside
        the 3-second hook window). A start with dead air is trimmed forward; a start that lands right on the line is
        given the same short lead-in, so no first syllable is clipped. Never into the tail of an earlier line, never
        into the previous scene or the previous segment, never outside the length limits. Returns segments moved."""
        moved = 0
        for c in cands:
            for k, sg in enumerate(c.segments):
                idx = self.index[sg.episode]
                beat = next((e for e in idx.events
                             if sg.start_sec - 1e-6 <= e.start_sec <= sg.start_sec + config.HOOK_WINDOW_SEC + 1e-6
                             and (e.track == "DIALOGUE" or (e.track in ("ACTION", "CAMERA")
                                                            and KINETIC_ACTION_RX.search(e.text)))), None)
                if beat is None:
                    continue
                new = max(0.0, round(beat.start_sec - pre_roll, 2))
                # an earlier line still speaking, or a shot from another scene, bounds the lead-in
                stops = [d.end_sec for d in idx.dialogue if d is not beat and new < d.end_sec <= beat.start_sec + 1e-6]
                stops += [e.end_sec for e in idx.events if e.scene_id != beat.scene_id
                          and new < e.end_sec <= beat.start_sec + 1e-6]
                if any(e.scene_id != beat.scene_id and e.start_sec <= new < e.end_sec - 1e-6 for e in idx.events):
                    stops.append(beat.start_sec)
                if stops:
                    new = round(max(stops), 2)
                prev = c.segments[k - 1] if k else None
                if prev is not None and prev.episode == sg.episode:
                    new = max(new, round(prev.end_sec, 2))
                delta = sg.start_sec - new          # > 0: lead-in added, < 0: dead air trimmed
                if abs(delta) < 0.05 or not (self.min_dur - 1e-6 <= c.total_duration_sec + delta <= self.max_dur + 1e-6):
                    continue
                c.signals.setdefault("tightened", []).append(
                    f"Ep{sg.episode} start {sg.start_timestamp} -> {sec_to_ts(new)} ({beat.start_ts} - "
                    f"{beat.start_sec - new:.1f} s)")
                sg.start_sec, sg.start_timestamp = new, sec_to_ts(new)
                sg.anchor_timestamp, sg.pre_roll_sec = beat.start_ts, round(beat.start_sec - new, 2)
                moved += 1
        return moved

    # ---------------------------------------------------------------- Type B
    def intra(self, ep: int) -> List[Candidate]:
        idx = self.index[ep]
        lo1, hi1 = config.INTRA_SEG1
        lo2, hi2 = config.INTRA_SEG2
        # Segment 1 = hook/setup: strong opening + a strong "button" line before the jump cut
        seg1: List[Tuple[float, float, float]] = []
        for s, h in self._good_starts(ep, 12):
            for e, _ in idx.clean_ends:
                if lo1 <= e - s <= hi1:
                    seg1.append((h + 0.35 * self.cliff_at(ep, e), s, e))
        seg1 = sorted(seg1, reverse=True)[:10]
        # Segment 2 = climax: re-hooks on a sharp line and ends on the strongest cliffhanger
        seg2: List[Tuple[float, float, float]] = []
        rehooks = self._good_starts(ep, 30, REHOOK_MIN)
        for e, c in self._good_ends(ep, 14):
            for s, h in rehooks:
                if lo2 <= e - s <= hi2:
                    seg2.append((c + 0.3 * h, s, e))
        seg2 = sorted(seg2, reverse=True)[:12]
        lo_t, hi_t = config.INTRA_TOTAL
        cands: List[Candidate] = []
        for _, s1, e1 in seg1:
            for _, s2, e2 in seg2:
                if s2 - e1 < config.INTRA_MIN_SKIP:
                    continue
                total = (e1 - s1) + (e2 - s2)
                if not (max(lo_t, self.min_dur) <= total <= min(hi_t, self.max_dur)):
                    continue
                ctx = self._ctx([(ep, s1, e1), (ep, s2, e2)])
                bs = score_blind(ctx, self.title_toks)
                cont = self._ctx([(ep, s1, e2)])
                skip_gain = pacing_score(ctx)[0] - pacing_score(cont)[0]
                spk1 = {d.speaker for d in ctx.segments[0].dialogue}
                spk2 = {d.speaker for d in ctx.segments[1].dialogue}
                coherence = 0.4 if spk1 & spk2 else -0.6
                bonus, notes = self.stage2_bonus(idx, e2)
                sel = (composite(bs.scores, self.weights) + 1.5 * max(skip_gain, 0) + coherence + bonus
                       + self._duration_pref(total))
                cands.append(Candidate(candidate_id="", promo_type="multi_segment_intra",
                                       segments=[self._seg(idx, s1, e1, "hook_setup"),
                                                 self._seg(idx, s2, e2, "climax")],
                                       signals={"stage2_notes": notes + [f"skips {s2 - e1:.1f}s of middle material"],
                                                "blind": bs.scores, "tags": bs.tags, "skip_pacing_gain": round(skip_gain, 2),
                                                "diagnostics": bs.diagnostics},
                                       arc_tags=bs.tags, selection_score=round(sel, 3)))
        self.stats.windows_evaluated += len(cands)
        kept = nms(cands, 0.5, config.MAX_INTRA_PER_EP)
        for i, c in enumerate(kept, 1):
            c.candidate_id = f"PROMO_EP{ep:02d}_INTRA_{i:02d}"
        self.stats.intra_kept += len(kept)
        return kept

    # ---------------------------------------------------------------- Type C
    def _entities(self, ep: int, s: float, e: float) -> Set[str]:
        idx = self.index[ep]
        ents: Set[str] = set()
        evs = idx.events_in(s, e)
        text = " ".join(x.text for x in evs).lower()
        for x in evs:
            if x.track == "DIALOGUE":
                ents.add(x.speaker)
        for name in self.profile.core_cast:
            first = name.split()[-1].lower()
            if re.search(r"\b" + re.escape(first) + r"\b", text):
                ents.add(name)
        return ents

    def _pool(self, ep: int, lo: float, hi: float, want: Set[str], kind: str, k: int = 4
              ) -> List[Tuple[float, float, float, List[str], Set[str]]]:
        idx = self.index[ep]
        out = []
        if kind == "setup":
            starts = self._good_starts(ep, 14)
            for s, h in starts:
                for e, _ in idx.clean_ends:
                    if lo <= e - s <= hi:
                        hits = trope_hits(" ".join(x.text for x in idx.events_in(s, e)))
                        tags = [t for t in hits if t in want]
                        if not tags:
                            continue
                        strength = max(TROPE_WEIGHTS[t] * min(1, hits[t] / 2) for t in tags)
                        out.append((h + 0.5 * self.cliff_at(ep, e) + 3 * strength, s, e, tags,
                                    self._entities(ep, s, e)))
        else:
            rehooks = self._good_starts(ep, 30, REHOOK_MIN)
            for e, c in self._good_ends(ep, 14):
                for s, h in rehooks:
                    if lo <= e - s <= hi:
                        hits = trope_hits(" ".join(x.text for x in idx.events_in(s, e)))
                        tags = [t for t in hits if t in want]
                        if not tags:
                            continue
                        strength = max(TROPE_WEIGHTS[t] * min(1, hits[t] / 2) for t in tags)
                        out.append((c + 0.3 * h + 3 * strength, s, e, tags, self._entities(ep, s, e)))
        out.sort(key=lambda x: -x[0])
        dedup: List[Tuple[float, float, float, List[str], Set[str]]] = []
        for item in out:
            if all(_iou((item[1], item[2]), (d[1], d[2])) < 0.3 for d in dedup):
                dedup.append(item)
            if len(dedup) >= k:
                break
        return dedup

    HUMBLED = ("desperate", "pleading", "begging", "sobbing", "weeping", "hysterical", "terrified", "trembling",
               "shaken", "denial", "broken", "defeated", "panicked", "crying")
    HUMBLED_ACTION = re.compile(r"\b(begs?|begging|pleads?|pleading|kneels?|kneeling|grovels?|grovelling|sobs?|"
                                r"cowers?|cowering|collapses?|on (?:her|his) knees|dragged away|in chains|arrested)\b", re.I)

    def _humbled_antagonists(self, ep: int, s: float, e: float) -> Set[str]:
        """Recurring antagonists shown begging / defeated inside the window (reads as 'payback'): a pleading
        speaker, or a recurring character named in an action line that shows them begging/kneeling/arrested.
        Reading action lines keeps the guard working when the speaker label is a description, not a name."""
        idx = self.index[ep]
        villains = self.profile.core_cast - self.profile.premise_cast
        out: Set[str] = set()
        for x in idx.events_in(s, e):
            if x.track == "DIALOGUE":
                if x.speaker not in villains:
                    continue
                tags = " ".join(x.emotion_tags).lower()
                if "plea" in idx.feat[x.idx].acts or any(h in tags for h in self.HUMBLED):
                    out.add(x.speaker)
            elif x.track == "ACTION" and self.HUMBLED_ACTION.search(x.text):
                for v in villains:
                    if re.search(r"\b" + re.escape(v.split()[-1]) + r"\b", x.text, re.I):
                        out.add(v)
        return out

    def cross(self, villain_guard: bool = True) -> List[Candidate]:
        """villain_guard: an antagonist shown begging/defeated in the payoff must already be in the setup.
        Without it the stitcher can cut from villain A's cruelty to villain B's punishment - a cold viewer
        reads that as A's comeuppance (a payoff the show never delivers)."""
        self.villain_swaps: List[Candidate] = []
        lo_s, hi_s = config.CROSS_SETUP
        lo_p, hi_p = config.CROSS_PAYOFF
        lo_t, hi_t = config.CROSS_TOTAL
        eps = sorted(self.index)
        setups = {ep: self._pool(ep, lo_s, hi_s, SETUP_TAGS, "setup") for ep in eps}
        payoffs = {ep: self._pool(ep, lo_p, hi_p, PAYOFF_TAGS, "payoff") for ep in eps}
        cands: List[Candidate] = []
        for e1 in eps:
            for sc1, s1, t1, tags1, ents1 in setups[e1]:
                for e2 in eps:
                    if e2 - e1 < config.CROSS_MIN_EP_GAP:
                        continue
                    for sc2, s2, t2, tags2, ents2 in payoffs[e2]:
                        total = (t1 - s1) + (t2 - s2)
                        if not (max(lo_t, self.min_dur) <= total <= min(hi_t, self.max_dur)):
                            continue
                        shared = (ents1 & ents2) & self.profile.core_cast
                        if not (shared & self.profile.protagonists) or len(shared) < 2:
                            continue      # arc must follow the heroine AND a recurring ally/antagonist
                        new_villains = self._humbled_antagonists(e2, s2, t2) - ents1
                        ctx = self._ctx([(e1, s1, t1), (e2, s2, t2)])
                        bs = score_blind(ctx, self.title_toks)
                        arc = f"{tags1[0]}->{tags2[0]}"
                        sel = (composite(bs.scores, self.weights) + 0.8 * (len(shared) - 1)
                               + 1.5 * TROPE_WEIGHTS[tags1[0]] + 1.5 * TROPE_WEIGHTS[tags2[0]]
                               + self._duration_pref(total))
                        i1, i2 = self.index[e1], self.index[e2]
                        cand = Candidate(
                            candidate_id="", promo_type="multi_segment_cross",
                            segments=[self._seg(i1, s1, t1, "inciting_wound"), self._seg(i2, s2, t2, "payoff_tease")],
                            signals={"stage2_notes": [f"shared characters: {', '.join(sorted(shared))}"],
                                     "blind": bs.scores, "tags": bs.tags, "setup_tags": tags1, "payoff_tags": tags2,
                                     "new_villains_in_payoff": sorted(new_villains), "diagnostics": bs.diagnostics},
                            arc=arc, arc_tags=sorted(set(tags1 + tags2)), selection_score=round(sel, 3))
                        if new_villains and villain_guard:
                            self.villain_swaps.append(cand)
                            continue
                        cands.append(cand)
        self.stats.windows_evaluated += len(cands)
        # greedy diverse selection: each segment once, <=2 uses per episode per role, <=4 per arc
        cands.sort(key=lambda c: -c.selection_score)
        kept: List[Candidate] = []
        used_setup: Dict[int, int] = {}
        used_payoff: Dict[int, int] = {}
        used_arc: Dict[str, int] = {}
        used_pairs: Set[Tuple[int, int]] = set()
        for c in cands:
            a, b = c.segments
            if used_setup.get(a.episode, 0) >= 2 or used_payoff.get(b.episode, 0) >= 2:
                continue
            if used_arc.get(c.arc or "", 0) >= 4 or (a.episode, b.episode) in used_pairs:
                continue
            if any(_span_iou(c, k) >= 0.3 for k in kept):
                continue
            kept.append(c)
            used_setup[a.episode] = used_setup.get(a.episode, 0) + 1
            used_payoff[b.episode] = used_payoff.get(b.episode, 0) + 1
            used_arc[c.arc or ""] = used_arc.get(c.arc or "", 0) + 1
            used_pairs.add((a.episode, b.episode))
            if len(kept) >= config.MAX_CROSS_TOTAL:
                break
        counter: Dict[Tuple[int, int], int] = {}
        for c in kept:
            a, b = c.segments
            counter[(a.episode, b.episode)] = counter.get((a.episode, b.episode), 0) + 1
            c.candidate_id = f"PROMO_ARC_EP{a.episode:02d}_EP{b.episode:02d}_{counter[(a.episode, b.episode)]:02d}"
        self.stats.cross_kept = len(kept)
        return kept

    # ---------------------------------------------------------------- driver
    def generate_all(self, include_intra: bool = True, include_cross: bool = True) -> List[Candidate]:
        out: List[Candidate] = []
        for ep in sorted(self.index):
            out.extend(self.continuous(ep))
            out.extend(self.premise_cuts(ep))
            if include_intra:
                out.extend(self.intra(ep))
        if include_cross and len(self.index) > 1:
            out.extend(self.cross())
        out.extend(self.premise_montages())
        return out

    # ---------------------------------------------------------------- snap-back trimmer
    def snap_trim(self, ep: int, s: float, e: float, max_lead: float = 15.0, max_tail: float = 8.0
                  ) -> Tuple[float, float, List[str]]:
        """Tighten an arbitrary (e.g. naive scene-aligned) window: skip slow-burn lead-ins and trim
        post-climax release / fade-to-black so the cut lands on the peak."""
        idx = self.index[ep]
        log: List[str] = []
        starts = [t for t, _ in idx.clean_starts]
        ends = [t for t, _ in idx.clean_ends]
        h0, ok0 = self.hook_at(ep, s) if s in starts else (0.0, False)
        if not ok0 or h0 < HOOK_MIN:
            opts = [(self.hook_at(ep, t)[0], t) for t in starts if s <= t <= s + max_lead and self.hook_at(ep, t)[1]]
            if opts:
                h, t = max(opts)
                if t != s:
                    log.append(f"start {s:.2f}s -> {t:.2f}s (hook {h0:.1f} -> {h:.1f})")
                    s = t
        if e > idx.transition_start + 1e-6:
            valid = [t for t in ends if t <= idx.transition_start + 1e-6]
            if valid:
                log.append(f"end {e:.2f}s -> {valid[-1]:.2f}s (removed fade/cut-to-black)")
                e = valid[-1]
        if e not in ends:
            prior = [t for t in ends if t <= e]
            if prior:
                e = prior[-1]
        c0 = self.cliff_at(ep, e)
        opts = [(self.cliff_at(ep, t) - 0.05 * (e - t), t) for t in ends
                if e - max_tail <= t <= e and t - s >= self.min_dur]
        if opts:
            c, t = max(opts)
            if t < e and self.cliff_at(ep, t) >= c0 + 0.75:
                log.append(f"end {e:.2f}s -> {t:.2f}s (cliff {c0:.1f} -> {self.cliff_at(ep, t):.1f})")
                e = t
        if e - s > self.max_dur:
            cap = [t for t in ends if t - s <= self.max_dur]
            if cap:
                best = max(cap, key=lambda t: self.cliff_at(ep, t) if t - s >= self.min_dur else -1)
                log.append(f"end {e:.2f}s -> {best:.2f}s (60s cap)")
                e = best
        return s, e, log
