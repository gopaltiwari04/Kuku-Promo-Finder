"""Typed records shared by every stage (plain dataclasses: zero dependencies, JSON-friendly)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

TRACKS = ("CAMERA", "ACTION", "DIALOGUE", "SOUND_FX")
PROMO_TYPES = ("continuous", "multi_segment_intra", "multi_segment_cross")
SCORE_KEYS = ("hook_strength", "cliffhanger_tension", "standalone_clarity",
              "pacing_density", "trope_virality", "spoiler_risk")


def sec_to_ts(sec: float) -> str:
    """Format seconds as MM:SS.ss (the screenplay's own format)."""
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


@dataclass
class ScriptEvent:
    idx: int
    track: str                    # CAMERA | ACTION | DIALOGUE | SOUND_FX
    start_ts: str                 # verbatim from the .txt, e.g. "00:11.30"
    end_ts: str
    start_sec: float
    end_sec: float
    text: str                     # visual description / spoken words (quotes stripped)
    scene_id: str
    header_raw: str = ""
    speaker_raw: str = ""         # label as written, e.g. "THE MASKED STRANGER"
    speaker: str = ""             # canonical entity, e.g. "MAYA"
    emotion_tags: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)   # O.S., V.O., Continues
    line_no: int = 0

    @property
    def duration(self) -> float:
        return round(self.end_sec - self.start_sec, 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = self.duration
        return d


@dataclass
class Scene:
    scene_id: str
    heading: str
    start_ts: str
    end_ts: str
    start_sec: float
    end_sec: float


@dataclass
class Episode:
    episode_number: int
    filename: str
    path: str
    logline: str
    continuity_notes: str
    scenes: List[Scene]
    events: List[ScriptEvent]
    duration_sec: float
    parse_warnings: List[str] = field(default_factory=list)
    timestamp_strings: List[str] = field(default_factory=list)   # every verbatim timestamp in the file

    @property
    def dialogue(self) -> List[ScriptEvent]:
        return [e for e in self.events if e.track == "DIALOGUE"]


@dataclass
class Segment:
    episode: int
    start_timestamp: str
    end_timestamp: str
    start_sec: float
    end_sec: float
    role: str = "main"            # main | hook_setup | climax | inciting_wound | payoff_tease | premise_*
    anchor_timestamp: str = ""    # tight start: the verbatim block the cut leans on ...
    pre_roll_sec: float = 0.0     # ... and how far before it the clip starts

    @property
    def duration_sec(self) -> float:
        # Wall-clock only. Never the sum of overlapping CAMERA/ACTION/DIALOGUE/SFX blocks.
        return round(self.end_sec - self.start_sec, 2)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "episode": self.episode,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_sec": self.duration_sec,
            "role": self.role,
        }
        if self.pre_roll_sec:
            d["anchor_timestamp"], d["pre_roll_sec"] = self.anchor_timestamp, self.pre_roll_sec
        return d


def total_duration(segments: List[Segment]) -> float:
    return round(sum(s.end_sec - s.start_sec for s in segments), 2)


@dataclass
class Candidate:
    candidate_id: str
    promo_type: str
    segments: List[Segment]
    signals: Dict[str, Any] = field(default_factory=dict)   # deterministic Stage-2 features
    arc: Optional[str] = None
    arc_tags: List[str] = field(default_factory=list)
    selection_score: float = 0.0

    @property
    def total_duration_sec(self) -> float:
        return total_duration(self.segments)

    @property
    def episodes(self) -> List[int]:
        out: List[int] = []
        for s in self.segments:
            if s.episode not in out:
                out.append(s.episode)
        return out


@dataclass
class Promo:
    promo_id: str
    promo_type: str
    segments: List[Dict[str, Any]]
    total_duration_sec: float
    hook: str
    cliffhanger: str
    why_it_works: str
    scores: Dict[str, float]
    hook_timestamp: str = ""
    cliffhanger_timestamp: str = ""
    episodes: List[int] = field(default_factory=list)
    arc: Optional[str] = None
    arc_tags: List[str] = field(default_factory=list)
    composite_score: float = 0.0
    wave: int = 4
    wave_label: str = ""
    wave_reason: str = ""
    post_reason: str = ""
    variant_of: str = ""
    score_rank: int = 0
    post_order: int = 0
    flags: List[str] = field(default_factory=list)
    needs_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    scored_by: str = "heuristic"
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCase:
    case_id: str
    failure_mode: str
    category: str                 # where_ai_got_it_wrong | where_tool_picks_badly | guardrail_caught
    title: str
    episode: Optional[int]
    naive_pick: Dict[str, Any]
    guarded_pick: Dict[str, Any]
    evidence: List[str]
    metrics: Dict[str, Any]
    fix: str
    residual_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
