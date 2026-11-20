"""Show profile - who the story is about, learned from the screenplays instead of hard-coded.

The pipeline needs to know three things about a show:
  * the lead (protagonist)            - cross-episode promos must follow her/him
  * the recurring cast                - a stitched "setup -> payback" must share recurring characters
  * the title character(s)            - "premise cast" (never counted as old lore; the title already sells them)

All three are derived from the episode files of whatever show is loaded:
  * named characters = speaker labels whose name also appears as a capitalised word in the episode headers or
    dialogue (role words like "Nobleman", "Priest", "Guard" are excluded)
  * presence = episodes where the character speaks or is named
  * lead = most lines x most episodes;  recurring cast = present in >= 10 % of episodes (min 3)
  * title character = the lead's most frequent scene partner, boosted when episode summaries mention them next to a
    word from the show title

Descriptor labels ("THE MASKED STRANGER", "WOMAN IN RED COAT") are not merged by code: matching them to a named
character is unreliable from text alone, and a wrong merge would blind the false-payback guard. They are resolved
by (1) an optional human-reviewed file profiles/<show>.json, or (2) one model call when a live provider is used.
Without either, the tool still works: descriptor speakers simply stay separate characters.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import config
from .schema import Episode

HONORIFICS = ("LADY ", "LORD ", "SIR ", "DAME ", "MISS ", "MR. ", "MRS. ", "MR ", "MRS ")
# generic English role, rank and description words: never a character's proper name. Used only to tell a speaker
# label like "NIGHT NURSE" from a name; it does not score anything. Show-specific descriptions ("LADY IN A GREY
# COAT") are learned per show by descriptor_words(). Extend per language.
ROLE_WORDS = {
    "lady", "lord", "sir", "dame", "miss", "mr", "mrs", "the", "a", "an", "of", "in", "with", "and",
    "duke", "duchess", "dowager", "king", "queen", "prince", "princess", "earl", "count", "countess", "baron",
    "father", "mother", "brother", "sister", "son", "daughter", "uncle", "aunt", "grandmother", "grandfather",
    "elder", "royal", "guard", "guards", "maid", "noble", "nobleman", "noblewoman", "blond", "blonde", "mysterious",
    "bearded", "scarred", "high", "priest", "crown", "official", "matriarch", "former", "groom", "bride",     "stranger", "man", "woman", "boy", "girl", "thug", "old", "young", "tall", "third", "bald",
    "peasant", "judge", "constable", "police", "officer", "informant", "housekeeper", "butler", "footman",
    "townsman", "driver", "loyalist", "fighter", "scout", "sentry", "villager", "aristocratic",
    "conspirator", "voice", "off", "screen", "seamstress", "doctor", "nurse", "servant", "soldier", "captain",
    "general", "assistant", "secretary", "ceo", "boss", "manager", "waiter", "reporter", "crowd", "all", "both",
    "first", "second", "episode", "series", "scene", "chapter",
}
_NAME_RX = re.compile(r"\b([A-Z][a-z]{2,})\b")


@dataclass
class ShowProfile:
    show_title: str
    protagonists: Set[str] = field(default_factory=set)
    premise_cast: Set[str] = field(default_factory=set)
    core_cast: Set[str] = field(default_factory=set)
    # descriptor label -> [(first_ep, last_ep, canonical name)]
    aliases: Dict[str, List[Tuple[int, int, str]]] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def canonical(self, label: str, episode: int, known_labels: Optional[Set[str]] = None) -> str:
        label = label.upper().strip()
        for lo, hi, canon in self.aliases.get(label, []):
            if lo <= episode <= hi:
                return canon
        if known_labels:
            for h in HONORIFICS:
                if label.startswith(h) and label[len(h):] in known_labels:
                    return label[len(h):]
        return label

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("protagonists", "premise_cast", "core_cast"):
            d[k] = sorted(d[k])
        d["aliases"] = {k: [list(x) for x in v] for k, v in self.aliases.items()}
        return d


def profile_slug(show_title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", show_title.lower()).strip("-")[:80] or "show"


def _key_name(label: str) -> str:
    """'LADY ROSE' -> 'Rose', 'ETHAN'S MOTHER' -> '' (relational label), 'MARCUS' -> 'Marcus'."""
    if label.endswith("'S MOTHER") or label.endswith("'S FATHER") or "'S " in label:
        return ""
    toks = [t for t in re.findall(r"[A-Z][A-Z'\-]+", label) if t.lower().strip("'") not in ROLE_WORDS]
    return toks[-1].title() if toks else ""


def _known_names(episodes: Sequence[Episode]) -> Set[str]:
    """Capitalised words used like names in headers/dialogue (role words excluded)."""
    text = " ".join(ep.logline + " " + ep.continuity_notes for ep in episodes)
    text += " " + " ".join(e.text for ep in episodes for e in ep.events if e.track == "DIALOGUE")
    c = Counter(w for w in _NAME_RX.findall(text) if w.lower() not in ROLE_WORDS)
    return {w for w, n in c.items() if n >= 2}


def _title_words(show_title: str) -> Set[str]:
    stop = {"married", "marry", "with", "that", "this", "from", "into", "after", "before", "when", "was", "were",
            "her", "his", "the", "and", "who", "what", "broken", "secret", "my", "our", "your"}
    return {w for w in re.findall(r"[a-z]+", show_title.lower()) if len(w) > 3 and w not in stop}


def build_profile(episodes: Sequence[Episode], show_title: str) -> ShowProfile:
    eps = list(episodes)
    n_eps = max(len(eps), 1)
    raw_labels = Counter(e.speaker_raw.upper().strip() for ep in eps for e in ep.events if e.track == "DIALOGUE")
    known_labels = set(raw_labels)
    names = _known_names(eps)
    prof = ShowProfile(show_title=show_title)
    # canonical label per raw label (honorific merge only), then named entities
    canon_of = {lab: prof.canonical(lab, 1, known_labels) for lab in raw_labels}
    lines: Counter = Counter()
    for lab, n in raw_labels.items():
        lines[canon_of[lab]] += n
    named: Dict[str, str] = {}          # canonical label -> key name
    for c in lines:
        k = _key_name(c)
        if k and k in names:
            named[c] = k
    # presence: episodes where the character speaks or is named anywhere in the episode text
    presence: Dict[str, Set[int]] = defaultdict(set)
    for ep in eps:
        txt = " ".join(e.text for e in ep.events)
        spoken_here = {canon_of.get(e.speaker_raw.upper().strip()) for e in ep.events if e.track == "DIALOGUE"}
        for c, k in named.items():
            if c in spoken_here or re.search(r"\b" + re.escape(k) + r"\b", txt):
                presence[c].add(ep.episode_number)
    if not named:
        prof.sources["all"] = "no named characters found - cast-dependent features are disabled"
        return prof
    lead = max(named, key=lambda c: (len(presence[c]) * math.log1p(lines[c]), lines[c], c))
    prof.protagonists = {lead}
    need = max(3, math.ceil(0.10 * n_eps)) if n_eps >= 5 else 1
    prof.core_cast = {c for c in named if len(presence[c]) >= need and lines[c] >= 3} | {lead}
    # title character: lead's most frequent scene partner, boosted by summary sentences that name them next to a title word
    tw = _title_words(show_title)
    header_sents = [s for ep in eps for s in re.split(r"(?<=[.!?])\s+", ep.logline + " " + ep.continuity_notes)]
    title_hits: Counter = Counter()
    for s in header_sents:
        if any(w in s.lower() for w in tw):
            for c, k in named.items():
                if re.search(r"\b" + re.escape(k) + r"\b", s):
                    title_hits[c] += 1
    co = {c: len(presence[c] & presence[lead]) for c in prof.core_cast if c != lead}
    premise = set()
    if co:
        best = max(co, key=lambda c: (co[c] + 3 * title_hits[c], lines[c], c))
        premise.add(best)
    prof.premise_cast = {lead} | premise
    prof.sources.update({"protagonists": "auto: most lines x most episodes",
                         "core_cast": f"auto: named and present in >= {need} episodes",
                         "premise_cast": "auto: lead + lead's most frequent scene partner (title-word mentions break ties)",
                         "aliases": "none (descriptor labels stay separate characters)"})
    prof.evidence = {"named_characters": {c: {"lines": lines[c], "episodes": len(presence[c])} for c in
                                          sorted(named, key=lambda x: -lines[x])},
                     "title_words": sorted(tw), "title_word_mentions": dict(title_hits),
                     "descriptor_labels": sorted(c for c in lines if c not in named and lines[c] >= 3)}
    return prof


def load_override(prof: ShowProfile, path: Path) -> bool:
    """Apply a human-reviewed profile file (aliases and, optionally, cast corrections)."""
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("aliases"):
        prof.aliases = {k.upper(): [(int(a), int(b), str(c).upper()) for a, b, c in v] for k, v in data["aliases"].items()}
        prof.sources["aliases"] = f"human-reviewed file {path.name}"
    for key in ("protagonists", "premise_cast", "core_cast"):
        if data.get(key):
            setattr(prof, key, {str(x).upper() for x in data[key]})
            prof.sources[key] = f"human-reviewed file {path.name}"
    return True


ALIAS_SYSTEM = """You map speaker labels in a screenplay to named characters. Some speakers are labelled by a \
description ("THE MASKED STRANGER") before the story names them. For each descriptor label you get a few of its lines \
and the shot descriptions around them, plus the list of named characters. Return ONLY JSON: \
{"aliases": [{"label": "<descriptor label>", "name": "<one of the named characters or null>", "first_ep": n, \
"last_ep": n}]}. Use null unless the evidence clearly identifies the character."""


def resolve_aliases_with_model(prof: ShowProfile, episodes: Sequence[Episode], adapter) -> int:
    """One model call: descriptor labels -> named characters. Only accepted when the answer names a known
    character; anything else stays unmerged. Returns the number of aliases added."""
    named = set(prof.evidence.get("named_characters", {}))
    descs = prof.evidence.get("descriptor_labels", [])
    if not (adapter is not None and adapter.is_live and named and descs):
        return 0
    blocks = []
    for lab in descs[:25]:
        samples = []
        for ep in episodes:
            evs = ep.events
            for i, e in enumerate(evs):
                if e.track == "DIALOGUE" and e.speaker_raw.upper().strip() == lab:
                    near = [v.text[:140] for v in evs[max(0, i - 2): i + 2] if v.track == "CAMERA"][:1]
                    samples.append(f"  Ep{ep.episode_number} \"{e.text[:120]}\"" + (f"  [shot: {near[0]}]" if near else ""))
                    break
            if len(samples) >= 3:
                break
        blocks.append(f"LABEL {lab}:\n" + "\n".join(samples))
    user = ("NAMED CHARACTERS: " + ", ".join(sorted(named)) + "\n\n" + "\n\n".join(blocks)
            + "\n\nReturn the JSON object only.")
    try:
        data, _hit = adapter.complete_json_cached(ALIAS_SYSTEM, user, 2048, tag="aliases")
    except Exception:
        return 0
    added = 0
    for a in (data or {}).get("aliases", []) if isinstance(data, dict) else []:
        lab, name = str(a.get("label", "")).upper().strip(), a.get("name")
        if lab in descs and isinstance(name, str) and name.upper().strip() in named:
            lo, hi = int(a.get("first_ep") or 1), int(a.get("last_ep") or 9999)
            prof.aliases.setdefault(lab, []).append((lo, hi, name.upper().strip()))
            added += 1
    if added:
        prof.sources["aliases"] = f"model ({adapter.label}): {added} descriptor labels mapped"
    return added


def apply_profile(episodes: Sequence[Episode], prof: ShowProfile) -> None:
    """Set every dialogue line's canonical speaker (aliases + 'LADY X' -> 'X')."""
    labels = {e.speaker_raw.upper().strip() for ep in episodes for e in ep.events if e.track == "DIALOGUE"}
    for ep in episodes:
        for e in ep.events:
            if e.track == "DIALOGUE":
                e.speaker = prof.canonical(e.speaker_raw, ep.episode_number, labels)


def profile_for(episodes: Sequence[Episode], show_title: str, adapter=None,
                profiles_dir: Path = config.PROJECT_ROOT / "profiles") -> ShowProfile:
    prof = build_profile(episodes, show_title)
    if not load_override(prof, profiles_dir / f"{profile_slug(show_title)}.json"):
        resolve_aliases_with_model(prof, episodes, adapter)
    apply_profile(episodes, prof)
    return prof


# ------------------------------------------------------------------------------------------- story sequences
# Places that are one place to a viewer (the wedding is "the church" whether the heading says nave, aisle or altar).
PLACE_GROUPS = [
    {"cathedral", "chapel", "church", "nave", "aisle", "altar", "sanctuary", "vestry", "temple"},
    {"cottage", "cabin", "hut", "hovel", "shack"},
    {"courtroom", "court", "tribunal", "dock"},
    {"office", "boardroom", "conference"},
    {"penthouse", "apartment", "flat", "loft", "suite"},
    {"hospital", "ward", "clinic", "infirmary"},
]
PLACE_STOP = {"int", "ext", "the", "of", "and", "a", "an", "s", "day", "night", "dusk", "dawn", "morning", "evening",
              "continuous", "later", "same", "time", "moments", "inside", "outside", "interior", "exterior"}
PICKS_UP_RX = re.compile(r"\b(?:picks up (?:directly|immediately)|follows directly|continues directly|"
                         r"immediately after|moments after|continuous)\b", re.I)


def place_key(heading: str) -> str:
    """'INT. CATHEDRAL AISLE – CONTINUOUS' -> 'place-group-0' (the church); 'INT. ASHBURY MANOR - DRAWING ROOM' ->
    'ashbury manor drawing room'. Two scenes are in the same place only if their keys are equal."""
    head = re.split(r"\s+[–—]\s+|\s+-\s+(?=(?:DAY|NIGHT|DUSK|DAWN|MORNING|EVENING|CONTINUOUS|LATER|MOMENTS)\b)",
                    heading.strip())[0]
    words = [w for w in re.findall(r"[a-z]+", head.lower()) if w not in PLACE_STOP and len(w) > 2]
    for i, grp in enumerate(PLACE_GROUPS):
        if set(words) & grp:
            return f"place-group-{i}"
    return " ".join(words)


def story_sequences(episodes: Sequence[Episode]) -> Dict[Tuple[int, str], str]:
    """Group scenes into story sequences: consecutive scenes in the same place that carry straight on (the heading
    says CONTINUOUS, or the episode notes say it picks up directly from the last one). The four-episode wedding is
    one sequence; the cottage the next day is another. Used so the posting plan is not one scene cut ten ways."""
    out: Dict[Tuple[int, str], str] = {}
    seq_n, prev_key = 0, None
    for ep in sorted(episodes, key=lambda e: e.episode_number):
        for i, sc in enumerate(ep.scenes):
            key = place_key(sc.heading) or prev_key
            carries_on = (bool(PICKS_UP_RX.search(sc.heading)) or
                          (i == 0 and bool(PICKS_UP_RX.search(ep.continuity_notes))) or i > 0)
            if seq_n == 0 or key != prev_key or not carries_on:
                seq_n += 1
            out[(ep.episode_number, sc.scene_id)] = f"S{seq_n:03d}"
            prev_key = key
    return out


def descriptor_words(episodes: Sequence[Episode]) -> Set[str]:
    """Words in speaker labels that the screenplay also uses as ordinary lowercase words ('the tall waiter',
    'a stranger in a grey coat'): descriptions, not names. Learned per show instead of a hand-written list."""
    label_words = {w for ep in episodes for e in ep.events if e.track == "DIALOGUE"
                   for w in re.findall(r"[A-Z][A-Z']+", e.speaker_raw)}
    visual = " ".join(e.text for ep in episodes for e in ep.events if e.track != "DIALOGUE")
    lower = Counter(re.findall(r"(?<![.!?]\s)\b([a-z][a-z']+)\b", visual))
    out = {w.title() for w in label_words if lower[w.lower()] >= 2}
    return out | {w.title() for w in ROLE_WORDS}
