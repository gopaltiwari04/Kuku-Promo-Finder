"""Stage 1 - deterministic multi-track screenplay parser + wall-clock timeline slicer (100% code).

Key rule: CAMERA / ACTION / DIALOGUE / SOUND FX blocks run on *parallel, overlapping* tracks.
A clip's duration is ALWAYS wall-clock (t_end - t_start), never the sum of block durations.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import config
from .lexicons import (CLOSEUP_RX, CONTINUES_RX, OPENER_DEPENDENT_RX, RESOLVE_SFX_RX, SFX_CLIFF_RX, SFX_IMPACT_RX,
                       STARTS_CONTINUATION_RX, TRANSITION_RX, anchors_in, dialogue_acts, dialogue_arousal,
                       kinetic_score)
from .schema import Episode, Scene, ScriptEvent

TS = r"(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)"
TS_LINE_RE = re.compile(r"^\s*Start\s*:\s*\(?\s*" + TS + r"\s*\)?\s*\|\s*End\s*:\s*\(?\s*" + TS + r"\s*\)?", re.I)
HEADER_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
SCENE_RE = re.compile(r"^\s*SCENE\s+(\d+[A-Z]?)\s*$", re.I)
SCENE_FIELD_RE = re.compile(r"^\s*Scene\s+(heading|start|end)\s*:\s*(.+?)\s*$", re.I)
META_RES = {
    "episode_number": re.compile(r"^\s*EPISODE NUMBER\s*:\s*(?:Episode\s*)?(\d+)", re.I | re.M),
    "logline": re.compile(r"^\s*EPISODE LOGLINE\s*:\s*(.+)$", re.I | re.M),
    "continuity_notes": re.compile(r"^\s*CONTINUITY NOTES\s*:\s*(.+)$", re.I | re.M),
}
QUALIFIER_RE = re.compile(r"^(?:o\.?\s?s\.?|v\.?\s?o\.?|cont'?d|continues|continued|off[- ]?screen|"
                          r"into (?:phone|radio)|on screen|filtered)$", re.I)
TRACK_ALIASES = {"CAMERA": "CAMERA", "SHOT": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SOUND_FX",
                 "SFX": "SOUND_FX", "SOUND": "SOUND_FX", "SOUND EFFECTS": "SOUND_FX", "MUSIC": "SOUND_FX"}
HONORIFICS = ("LADY ", "LORD ", "SIR ", "DAME ", "MISS ", "MR. ", "MRS. ")


def ts_to_sec(ts: str) -> float:
    """'MM:SS.ss' (or 'HH:MM:SS.ss') -> float seconds, exactly mm*60 + ss."""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return round(int(parts[0]) * 60.0 + float(parts[1]), 3)
    if len(parts) == 3:
        return round(int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2]), 3)
    raise ValueError(f"Bad timestamp: {ts!r}")


def parse_speaker_header(header: str) -> Tuple[str, List[str], List[str]]:
    """'BOSS (O.S.) (Firm)' -> ('BOSS', ['Firm'], ['O.S.'])."""
    h = header.strip().rstrip(":").strip()
    name = h.split("(")[0].strip().rstrip(":").strip()
    emotions: List[str] = []
    qualifiers: List[str] = []
    for paren in re.findall(r"\(([^)]*)\)", h):
        for tok in re.split(r"[,/;]", paren):
            tok = tok.strip()
            if not tok:
                continue
            (qualifiers if QUALIFIER_RE.match(tok) else emotions).append(tok)
    return name.upper(), emotions, qualifiers


def classify_header(header: str) -> str:
    key = header.strip().rstrip(":").strip().upper()
    return TRACK_ALIASES.get(key, "DIALOGUE")


def parse_text(text: str, filename: str = "<memory>", path: str = "") -> Episode:
    warnings: List[str] = []
    meta: Dict[str, str] = {}
    for k, rx in META_RES.items():
        m = rx.search(text)
        meta[k] = m.group(1).strip() if m else ""
    ep_num = int(meta["episode_number"]) if meta["episode_number"] else 0
    if not ep_num:
        m = config.EPISODE_FILE_RE.match(Path(filename).name)
        ep_num = int(m.group(1)) if m else 0
        warnings.append("EPISODE NUMBER header missing; used filename prefix")

    body = text
    if "--- SCRIPT START ---" in text:
        body = text.split("--- SCRIPT START ---", 1)[1]
    else:
        warnings.append("SCRIPT START marker missing; parsed whole file")
    if "--- SCRIPT END ---" in body:
        body = body.split("--- SCRIPT END ---", 1)[0]

    # line offset of body inside file (for line numbers)
    offset = text[: len(text) - len(body)].count("\n") if body in text else 0
    lines = body.splitlines()

    scenes: List[Scene] = []
    events: List[ScriptEvent] = []
    cur_scene: Optional[Dict[str, str]] = None
    i = 0
    scene_counter = 0

    def flush_scene():
        if cur_scene is None:
            return
        st = cur_scene.get("start", "")
        en = cur_scene.get("end", "")
        try:
            scenes.append(Scene(cur_scene["id"], cur_scene.get("heading", ""), st, en,
                                ts_to_sec(st) if st else 0.0, ts_to_sec(en) if en else 0.0))
        except ValueError:
            warnings.append(f"bad scene timestamps in scene {cur_scene['id']}")

    while i < len(lines):
        line = lines[i]
        m_scene = SCENE_RE.match(line)
        if m_scene:
            flush_scene()
            scene_counter += 1
            cur_scene = {"id": m_scene.group(1).zfill(2)}
            i += 1
            continue
        m_field = SCENE_FIELD_RE.match(line)
        if m_field and cur_scene is not None:
            cur_scene[m_field.group(1).lower()] = m_field.group(2).strip()
            i += 1
            continue
        m_head = HEADER_RE.match(line)
        if m_head:
            header = m_head.group(1).strip()
            hdr_line = i
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            m_ts = TS_LINE_RE.match(lines[j]) if j < len(lines) else None
            if not m_ts:
                warnings.append(f"line {offset + hdr_line + 1}: block '{header}' has no Start/End line; skipped")
                i += 1
                continue
            k = j + 1
            text_lines: List[str] = []
            while k < len(lines):
                ln = lines[k]
                if HEADER_RE.match(ln) or SCENE_RE.match(ln) or TS_LINE_RE.match(ln) or SCENE_FIELD_RE.match(ln):
                    break
                if not ln.strip():
                    # a blank line ends the block unless more prose follows before the next header
                    nxt = k + 1
                    while nxt < len(lines) and not lines[nxt].strip():
                        nxt += 1
                    if nxt >= len(lines) or HEADER_RE.match(lines[nxt]) or SCENE_RE.match(lines[nxt]):
                        break
                else:
                    text_lines.append(ln.strip())
                k += 1
            start_ts, end_ts = m_ts.group(1), m_ts.group(2)
            track = classify_header(header)
            txt = " ".join(text_lines).strip()
            speaker_raw, emotions, quals = "", [], []
            if track == "DIALOGUE":
                speaker_raw, emotions, quals = parse_speaker_header(header)
                txt = txt.strip().strip('"').strip("“”").strip()
            s_sec, e_sec = ts_to_sec(start_ts), ts_to_sec(end_ts)
            if e_sec < s_sec:
                warnings.append(f"line {offset + j + 1}: end before start ({start_ts}>{end_ts}); swapped")
                s_sec, e_sec, start_ts, end_ts = e_sec, s_sec, end_ts, start_ts
            events.append(ScriptEvent(
                idx=len(events), track=track, start_ts=start_ts, end_ts=end_ts, start_sec=s_sec, end_sec=e_sec,
                text=txt, scene_id=(cur_scene or {"id": "01"})["id"], header_raw=header,
                speaker_raw=speaker_raw, speaker=speaker_raw, emotion_tags=emotions, qualifiers=quals,
                line_no=offset + hdr_line + 1))
            i = k
            continue
        i += 1
    flush_scene()

    if not scenes and events:
        warnings.append("no SCENE blocks; synthesised one scene")
        scenes.append(Scene("01", "", events[0].start_ts, max(events, key=lambda e: e.end_sec).end_ts,
                            events[0].start_sec, max(e.end_sec for e in events)))
    # stable chronological order (8 blocks in the corpus start slightly before their predecessor)
    events.sort(key=lambda e: (e.start_sec, e.idx))
    for n, e in enumerate(events):
        e.idx = n
    duration = max([s.end_sec for s in scenes] + [e.end_sec for e in events] + [0.0])
    ts_strings = sorted(set(re.findall(r"\d{2}:\d{2}\.\d{2}", text)))
    return Episode(episode_number=ep_num, filename=Path(filename).name, path=path or filename,
                   logline=meta["logline"], continuity_notes=meta["continuity_notes"], scenes=scenes,
                   events=events, duration_sec=round(duration, 2), parse_warnings=warnings,
                   timestamp_strings=ts_strings)


def parse_file(path: Path) -> Episode:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return parse_text(text, filename=Path(path).name, path=str(path))


def _natural_key(p: Path):
    m = config.EPISODE_FILE_RE.match(p.name)
    return (int(m.group(1)) if m else 10 ** 6, p.name)


def load_corpus(input_dir: Path) -> Tuple[List[Episode], List[Dict[str, str]]]:
    """Parse every episode .txt in a folder (read-only). Returns (episodes, errors)."""
    files = sorted([p for p in Path(input_dir).iterdir() if p.is_file() and config._looks_like_episode(p)],
                   key=_natural_key)
    episodes: List[Episode] = []
    errors: List[Dict[str, str]] = []
    for f in files:
        try:
            ep = parse_file(f)
            if not ep.events:
                errors.append({"file": f.name, "error": "no timestamped blocks found"})
                continue
            episodes.append(ep)
        except Exception as exc:  # pragma: no cover - surfaced in verification report
            errors.append({"file": f.name, "error": repr(exc)})
    episodes.sort(key=lambda e: e.episode_number)
    resolve_entities(episodes)
    return episodes, errors


def resolve_entities(episodes: Sequence[Episode], profile=None) -> None:
    """Canonical speaker per dialogue line. Default: 'LADY X' -> 'X' when 'X' is also a speaker label.
    With a ShowProfile, its aliases (human-reviewed file or model) are applied too."""
    if profile is not None:
        from .show_profile import apply_profile
        apply_profile(episodes, profile)
        return
    labels: Set[str] = {e.speaker_raw.upper().strip() for ep in episodes for e in ep.events if e.track == "DIALOGUE"}
    for ep in episodes:
        for e in ep.events:
            if e.track == "DIALOGUE":
                e.speaker = canonical_speaker(e.speaker_raw, labels)


def canonical_speaker(label: str, known_labels: Optional[Set[str]] = None) -> str:
    label = label.upper().strip()
    if known_labels:
        for h in HONORIFICS:
            if label.startswith(h) and label[len(h):] in known_labels:
                return label[len(h):]
    return label


# ======================================================================================
# Per-episode index + timeline slicer
# ======================================================================================
@dataclass
class EventFeatures:
    arousal: float = 0.0           # dialogue arousal 0..5
    acts: Tuple[str, ...] = ()
    kinetic: float = 0.0           # visual/SFX physical intensity 0..3
    closeup: bool = False
    transition: bool = False
    sfx_impact: bool = False
    sfx_cliff: bool = False
    sfx_resolves: bool = False     # a triumphant / gentle score fading out
    continues: bool = False        # line trails off "..." and is continued by the next line
    is_continuation: bool = False  # line starts with "..."
    dependent_opener: bool = False # "Because..", "And..", "No.." - needs prior context
    anchors: Tuple[str, ...] = ()


class EpisodeIndex:
    """Pre-computed features + clean cut points for one episode."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.events = ep.events
        self.starts = [e.start_sec for e in self.events]
        self.dialogue = [e for e in self.events if e.track == "DIALOGUE"]
        self.feat: Dict[int, EventFeatures] = {}
        for e in self.events:
            f = EventFeatures()
            if e.track == "DIALOGUE":
                f.arousal = dialogue_arousal(e.text, e.emotion_tags)
                f.acts = dialogue_acts(e.text)
                f.anchors = tuple(sorted(anchors_in(e.text)))
                f.is_continuation = bool(STARTS_CONTINUATION_RX.match(e.text))
                f.dependent_opener = bool(OPENER_DEPENDENT_RX.match(e.text))
                f.continues = bool(CONTINUES_RX.search(e.text))
            else:
                f.kinetic = kinetic_score(e.text)
                f.closeup = e.track == "CAMERA" and bool(CLOSEUP_RX.search(e.text))
                f.transition = e.track == "CAMERA" and bool(TRANSITION_RX.match(e.text))
                if e.track == "SOUND_FX":
                    f.sfx_impact = bool(SFX_IMPACT_RX.search(e.text))
                    f.sfx_resolves = bool(RESOLVE_SFX_RX.search(e.text))
                    f.sfx_cliff = bool(SFX_CLIFF_RX.search(e.text)) and not f.sfx_resolves
                    f.kinetic = max(f.kinetic, 1.6 if f.sfx_impact else 0.0)
            self.feat[e.idx] = f
        # a "..." line is only a real continuation if the same speaker resumes with "..." soon after
        for a, b in zip(self.dialogue, self.dialogue[1:]):
            fa, fb = self.feat[a.idx], self.feat[b.idx]
            fa.continues = fa.continues and fb.is_continuation and (b.start_sec - a.end_sec) < 4.0
        if self.dialogue:
            self.feat[self.dialogue[-1].idx].continues = False
        self.transition_start = min([e.start_sec for e in self.events if self.feat[e.idx].transition] +
                                    [ep.duration_sec])
        self.clean_starts = self._clean_starts()
        self.clean_ends = self._clean_ends()

    # ---- cut points ---------------------------------------------------------------
    def _straddles_dialogue(self, t: float, tol: float = 0.05) -> bool:
        return any(d.start_sec + tol < t < d.end_sec - tol for d in self.dialogue)

    def _clean_starts(self) -> List[Tuple[float, str]]:
        """Block starts where no line of dialogue is mid-sentence and before any fade-to-black."""
        seen: Dict[float, str] = {}
        for e in self.events:
            t = e.start_sec
            if t >= self.transition_start - 1e-6 or t in seen:
                continue
            if self._straddles_dialogue(t):
                continue
            seen[t] = e.start_ts
        return sorted(seen.items())

    def _clean_ends(self) -> List[Tuple[float, str]]:
        """Block ends where no dialogue is cut mid-line, no '...' line is left hanging, before any fade."""
        seen: Dict[float, str] = {}
        for e in self.events:
            t = e.end_sec
            if t > self.transition_start + 1e-6 or t in seen:
                continue
            if self._straddles_dialogue(t):
                continue
            last = self.last_dialogue_before(t)
            if last is not None and self.feat[last.idx].continues and last.end_sec <= t:
                nxt = self.next_dialogue_after(last.end_sec - 1e-6, exclude=last.idx)
                if nxt is not None and nxt.start_sec >= t - 1e-6:
                    continue   # we would cut between "..." and its completion
            seen[t] = e.end_ts
        return sorted(seen.items())

    # ---- lookups ------------------------------------------------------------------
    def events_in(self, s: float, e: float) -> List[ScriptEvent]:
        """All events overlapping [s, e] (parallel tracks included)."""
        hi = bisect_right(self.starts, e - 1e-6)
        return [ev for ev in self.events[:hi] if ev.end_sec > s + 1e-6]

    def last_dialogue_before(self, t: float) -> Optional[ScriptEvent]:
        best = None
        for d in self.dialogue:
            if d.end_sec <= t + 1e-6:
                best = d
            else:
                break
        return best

    def next_dialogue_after(self, t: float, exclude: int = -1) -> Optional[ScriptEvent]:
        for d in self.dialogue:
            if d.start_sec >= t and d.idx != exclude:
                return d
        return None

@dataclass
class WindowSlice:
    episode: int
    start_sec: float
    end_sec: float
    events: List[ScriptEvent]
    duration: float
    hook_events: List[ScriptEvent]
    first_dialogue_offset: Optional[float]
    hook_dialogue_present: bool
    hook_max_arousal: float
    hook_kinetic: float
    max_dead_air: float
    dialogue_coverage: float
    anchor_terms: List[str]
    speakers: List[str]
    scenes: List[str]
    contained_block_sum: float = 0.0          # the WRONG number a naive summer would report
    extras: Dict[str, float] = field(default_factory=dict)


def _union_length(intervals: Iterable[Tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def slice_window(idx: EpisodeIndex, start_sec: float, end_sec: float, hook_sec: float = config.HOOK_WINDOW_SEC
                 ) -> WindowSlice:
    """Collect every overlapping CAMERA/ACTION/DIALOGUE/SOUND_FX event and compute window metrics."""
    evs = idx.events_in(start_sec, end_sec)
    dur = round(end_sec - start_sec, 2)
    hook = [e for e in evs if start_sec - 1e-6 <= e.start_sec <= start_sec + hook_sec]
    dlg = [e for e in evs if e.track == "DIALOGUE" and e.start_sec >= start_sec - 1e-6 and e.end_sec <= end_sec + 1e-6]
    first_off = min((d.start_sec - start_sec for d in dlg), default=None)
    hook_dlg = [d for d in dlg if d.start_sec <= start_sec + hook_sec]
    hook_ar = max((idx.feat[d.idx].arousal for d in hook_dlg), default=0.0)
    hook_kin = max((idx.feat[e.idx].kinetic for e in hook if e.track != "DIALOGUE"), default=0.0)
    # dead air: longest stretch with no dialogue and no high-salience physical action
    busy = [(max(d.start_sec, start_sec), min(d.end_sec, end_sec)) for d in dlg]
    busy += [(max(e.start_sec, start_sec), min(e.end_sec, end_sec)) for e in evs
             if e.track != "DIALOGUE" and idx.feat[e.idx].kinetic >= 2.0]
    busy.sort()
    gaps, cursor = [], start_sec
    for s, e in busy:
        if s > cursor:
            gaps.append(s - cursor)
        cursor = max(cursor, e)
    gaps.append(max(0.0, end_sec - cursor))
    coverage = _union_length([(max(d.start_sec, start_sec), min(d.end_sec, end_sec)) for d in dlg]) / max(dur, 1e-6)
    anchors = sorted({a for d in dlg for a in idx.feat[d.idx].anchors})
    speakers = []
    for d in dlg:
        if d.speaker not in speakers:
            speakers.append(d.speaker)
    scenes = []
    for e in evs:
        if e.scene_id not in scenes:
            scenes.append(e.scene_id)
    contained = sum(e.duration for e in evs if e.start_sec >= start_sec - 1e-6 and e.end_sec <= end_sec + 1e-6)
    return WindowSlice(episode=idx.ep.episode_number, start_sec=start_sec, end_sec=end_sec, events=evs,
                       duration=dur, hook_events=hook, first_dialogue_offset=first_off,
                       hook_dialogue_present=bool(hook_dlg), hook_max_arousal=round(hook_ar, 2),
                       hook_kinetic=round(hook_kin, 2), max_dead_air=round(max(gaps), 2),
                       dialogue_coverage=round(coverage, 3), anchor_terms=anchors, speakers=speakers,
                       scenes=scenes, contained_block_sum=round(contained, 2))

