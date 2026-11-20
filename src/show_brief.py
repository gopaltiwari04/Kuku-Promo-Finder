"""Show brief - what THIS show sells, learned per run. Nothing about a particular show is hard-coded.

  premise_setup  the situation the title sets up           "I Married a Poor Driver"
  premise_twist  the twist the title promises               "He Was a Billionaire"
  twist_terms    words a character says when the twist lands {"billionaire", "heir", "fortune"}
  setup_terms    words that state the setup                 {"married", "soldier"}
  themes         the show's story engines                   ["public humiliation", "framed for a crime", ...]
  stakes_terms   this show's own high-stakes vocabulary     learned from the script's threat/reveal lines

Two ways to fill it, same fields:
  * model    one cached call reads the title and a sample of the show's most intense lines (any provider)
  * offline  the title is split into its setup and twist clauses; stakes words are the words far more common in
             shouted, sobbing or threatening lines than in the show's dialogue as a whole
Every word a model proposes must actually occur in the screenplays, so the brief cannot invent vocabulary.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from .lexicons import ANCHOR_RX, TROPE_LABELS, dialogue_acts, emotion_arousal, trope_hits
from .schema import Episode
from .scoring import title_reveal_nouns

STOP = set("""
a about above after again against all almost also am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each else ever
every few for from further get gets getting give go goes going gone got had hadn't has hasn't have haven't having he
he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself just let let's like look made make many may me might more most much must mustn't my myself never no nor
not now of off on once one only or other ought our ours ourselves out over own really right said same say says see
shall shan't she she'd she'll she's should shouldn't so some still such take tell than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've thing things think this those though
through to too under until up upon us very want was wasn't we we'd we'll we're we've well were weren't what what's
when when's where where's whether which while who who's whom why why's will with without won't would wouldn't yes yet
you you'd you'll you're you've your yours yourself yourselves back come came know knew known even away yourself
anything everything nothing something someone anyone everyone nobody time day night today tonight long little enough
always another around came ever going here away truly perhaps upon already once again inside outside left right
told tells telling keep kept took taken gave given find found stand standing stay watch watching coming means mean
call called people women woman men man girl boy mistake permission place word words world life years year
""".split())
GENERIC_EMOTION = set("""please help stop wait sorry thank thanks love hate want need dear darling god lord sir madam
mister miss yes okay alright enough leave listen look come away never always""".split())
# common English words that carry no story-specific stake (verbs, adjectives, body parts, time words)
COMMON = set("""able afraid alive almost alone already angry anymore answer ask asked believe belong belongs best
better big body born bring brought care careful certain change chance clear close cold could course cried dark
dead deal death decide deserve deserved die died different done dream easy end ending eyes face fact fail fair fall
fault fear feel feeling felt fight final finally fine first forget forgive free friend full gave glad good great hand
hands happen happened happy hard head hear heard heart heavy hold home honest hope hurt idea important kind kill
known last late laugh learn least less lie life light live lived living lose lost loved loves loving lucky matter
meant mind minute moment move name need needed nice open order pain part past pay perfect plan play poor power
pretty proud prove question quiet ready real reason remember rest return safe saved save second seen sense
shame short show side simple sleep small soon speak stay still story strong sure talk tear tears thought together
tomorrow touch tried true trust truth try turn understand wanted warm wear whole wish wonder work worse worst wrong
young exactly really simply surely""".split())

BRIEF_SYSTEM = """You read the title and a sample of lines from a vertical microdrama (ReelShort / DramaBox style) \
and write a short brief that a promo editor will use to pick ads. Return ONLY JSON:
{"premise_setup": "<= 12 words: the situation the title sets up",
 "premise_twist": "<= 12 words: the reveal the title promises, or \\"\\" if it promises none",
 "twist_terms": ["up to 6 single lowercase words a character would SAY on screen when that twist is revealed"],
 "setup_terms": ["up to 6 single lowercase words that state the setup"],
 "themes": ["3 to 5 short story engines, e.g. public humiliation, framed for a crime, hidden identity"],
 "stakes_terms": ["12 to 25 single lowercase words that mark high stakes in THIS show: dangers, crimes, punishments, \
secrets, objects of power"]}
Use only words that appear in the lines you are given (the title counts). No character names."""


@dataclass
class ShowBrief:
    title: str
    premise_setup: str = ""
    premise_twist: str = ""
    twist_terms: List[str] = field(default_factory=list)
    setup_terms: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    stakes_terms: List[str] = field(default_factory=list)
    source: str = "title + screenplay statistics"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------- offline
def _words(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z'\-]{2,}", (text or "").lower())


def title_clauses(title: str) -> Dict[str, str]:
    """'I Married a Poor Driver. He Was a Billionaire' -> setup 'I Married a Poor Driver', twist 'He Was a
    Billionaire'."""
    parts = [p.strip() for p in re.split(r"[.!?;:—–]|\s-\s", title) if p.strip()]
    twist = next((p for p in parts if re.search(r"\b(?:is|was|are|were)\b", p, re.I)), "")
    setup = " ".join(p for p in parts if p != twist)
    if not setup:          # "My Secret Husband Is the CEO": the setup is what comes before "is"
        setup = re.split(r"\b(?:is|was|are|were)\b", twist or title, maxsplit=1, flags=re.I)[0].strip() if twist else title
    return {"setup": setup, "twist": twist}


def cast_tokens(episodes: Sequence[Episode]) -> Set[str]:
    """Lowercase name tokens of every speaker label (names are not stakes words)."""
    toks: Set[str] = set()
    for ep in episodes:
        for e in ep.events:
            if e.track == "DIALOGUE":
                toks |= {w.lower().strip("'") for w in re.findall(r"[A-Za-z']+", e.speaker_raw or "")}
    return toks


MIN_EPISODES_FOR_STAKES = 8


def learn_stakes_terms(episodes: Sequence[Episode], exclude: Iterable[str] = (), k: int = 24) -> List[str]:
    """Words that are much more frequent in high-stakes lines (a threat, accusation or reveal, or a furious /
    sobbing delivery) than in the show's dialogue overall, used in at least three episodes. Place words from the
    scene headings and proper names (capitalised mid-sentence) are left out: a church is a setting, not a stake.
    Under MIN_EPISODES_FOR_STAKES episodes the counts are too small to trust, so it returns nothing (the model
    brief can still supply them)."""
    if len(episodes) < MIN_EPISODES_FOR_STAKES:
        return []
    exclude = {w.lower() for w in exclude}
    exclude |= {w for ep in episodes for sc in ep.scenes for w in _words(sc.heading)}
    hi, allc = Counter(), Counter()
    eps_of: Dict[str, Set[int]] = defaultdict(set)
    for ep in episodes:
        for e in ep.events:
            if e.track != "DIALOGUE":
                continue
            toks = {re.sub(r"['\u2019]s$", "", w).strip("'-") for w in _words(e.text)}
            toks = {w for w in toks if len(w) > 3 and w not in STOP and w not in GENERIC_EMOTION and w not in COMMON
                    and w not in exclude}
            high = (emotion_arousal(e.emotion_tags) >= 2.5
                    or bool({"threat", "accusation", "reveal"} & set(dialogue_acts(e.text))))
            for w in toks:
                allc[w] += 1
                eps_of[w].add(ep.episode_number)
                if high:
                    hi[w] += 1
    caps = Counter(m.lower() for ep in episodes for e in ep.events if e.track == "DIALOGUE"
                   for m in re.findall(r"(?<=[a-z,;] )([A-Z][a-z]+)", e.text))      # mid-sentence capitals = names
    H, A = max(1, sum(hi.values())), max(1, sum(allc.values()))
    n_eps = len(episodes)
    min_hi, min_eps = (4, 3) if n_eps >= 10 else (2, min(2, n_eps))
    scored = []
    for w, n in allc.items():
        if hi[w] < min_hi or len(eps_of[w]) < min_eps:
            continue
        if dialogue_acts(w) or ANCHOR_RX.fullmatch(w) or caps[w] >= 0.5 * n:
            continue                      # already covered by the genre-general lexicons
        lift = ((hi[w] + 0.5) / H) / ((n + 0.5) / A)
        if lift < 1.4:
            continue
        scored.append((lift * math.log(1 + hi[w]), w))
    return [w for _, w in sorted(scored, reverse=True)[:k]]


def stem_variants(episodes: Sequence[Episode], nouns: Sequence[str], min_count: int = 2) -> List[str]:
    """Spoken forms of the title's twist noun found in this script ('heir' -> 'heirs', 'heiress'): words that
    start with the noun and occur at least twice in dialogue. Synonyms a stem cannot reach come from the model brief."""
    if not nouns:
        return []
    c = Counter(w for ep in episodes for e in ep.events if e.track == "DIALOGUE" for w in _words(e.text))
    stems = [n.lower() for n in nouns if len(n) >= 3]
    return sorted(w for w, k in c.items() if k >= min_count and w not in stems and not re.search(r"['\u2019]s$", w)
                  and any(w.startswith(s) and len(w) - len(s) <= 4 for s in stems))


def offline_brief(episodes: Sequence[Episode], title: str) -> ShowBrief:
    cl = title_clauses(title)
    twist = sorted(title_reveal_nouns(cl["twist"])) if cl["twist"] else []
    twist += [w for w in stem_variants(episodes, twist) if w not in twist]
    setup = [w for w in _words(cl["setup"]) if len(w) > 3 and w not in STOP][:6]
    text = " ".join(e.text for ep in episodes for e in ep.events if e.track == "DIALOGUE")
    tropes = trope_hits(text)
    themes = [TROPE_LABELS[k].lower() for k, _ in sorted(tropes.items(), key=lambda kv: -kv[1])[:4]]
    stakes = learn_stakes_terms(episodes, exclude=cast_tokens(episodes) | set(twist) | set(setup))
    return ShowBrief(title=title, premise_setup=cl["setup"], premise_twist=cl["twist"], twist_terms=twist[:8],
                     setup_terms=setup, themes=themes, stakes_terms=stakes)


# ------------------------------------------------------------------------------------------- model
def _sample_lines(episodes: Sequence[Episode], max_chars: int = 12000) -> str:
    """The opening of episode 1 (premise) plus the most intense lines across the show, in story order."""
    eps = sorted(episodes, key=lambda e: e.episode_number)
    lines: List[str] = []
    if eps:
        lines += [f"Ep{eps[0].episode_number} {d.speaker_raw}: {d.text}" for d in eps[0].dialogue[:30]]
    scored = []
    for ep in eps:
        for d in ep.dialogue:
            a = emotion_arousal(d.emotion_tags) + (1 if "!" in d.text else 0) + 0.5 * len(dialogue_acts(d.text))
            scored.append((a, ep.episode_number, d.start_sec, f"Ep{ep.episode_number} {d.speaker_raw}: {d.text}"))
    top = sorted(scored, key=lambda x: -x[0])[:160]
    lines += [s for *_, s in sorted(top, key=lambda x: (x[1], x[2]))]
    out, n = [], 0
    for s in lines:
        if n + len(s) > max_chars:
            break
        out.append(s)
        n += len(s) + 1
    return "\n".join(out)


def _clean_terms(raw: Any, corpus: str, limit: int, min_count: int = 1) -> List[str]:
    out: List[str] = []
    for w in raw if isinstance(raw, list) else []:
        w = str(w).lower().strip()
        if not re.fullmatch(r"[a-z][a-z'\-]{2,}(?: [a-z][a-z'\-]{2,})?", w) or w in STOP or w in out:
            continue
        if len(re.findall(r"\b" + re.escape(w) + r"\b", corpus)) >= min_count:
            out.append(w)
        if len(out) >= limit:
            break
    return out


def model_brief(episodes: Sequence[Episode], title: str, adapter, use_cache: bool = True) -> ShowBrief:
    """One cached model call. Every proposed word is checked against the screenplays; empty fields fall back to
    the offline brief, so a bad answer can only make the brief smaller, never wrong."""
    base = offline_brief(episodes, title)
    user = f"TITLE: {title}\n\nLINES:\n{_sample_lines(episodes)}\n\nReturn the JSON object only."
    data, _hit = adapter.complete_json_cached(BRIEF_SYSTEM, user, 2048, tag="show brief", use_cache=use_cache)
    corpus = (title + " " + " ".join(e.text for ep in episodes for e in ep.events)).lower()
    names = cast_tokens(episodes)
    stakes = [w for w in _clean_terms(data.get("stakes_terms"), corpus, 25, 2) if w not in names]
    twist = _clean_terms(data.get("twist_terms"), corpus, 6)
    setup = _clean_terms(data.get("setup_terms"), corpus, 6)
    themes = [str(t).strip().lower()[:40] for t in (data.get("themes") or []) if str(t).strip()][:5]
    return ShowBrief(title=title,
                     premise_setup=str(data.get("premise_setup") or base.premise_setup)[:120],
                     premise_twist=str(data.get("premise_twist") or base.premise_twist)[:120],
                     twist_terms=sorted(set(twist) | set(base.twist_terms)) if twist or base.twist_terms else [],
                     setup_terms=setup or base.setup_terms, themes=themes or base.themes,
                     stakes_terms=stakes or base.stakes_terms, source=f"model ({adapter.label})")


def build_brief(episodes: Sequence[Episode], title: str, adapter=None, use_cache: bool = True,
                log=None) -> ShowBrief:
    if adapter is not None and adapter.is_live:
        try:
            return model_brief(episodes, title, adapter, use_cache)
        except Exception as exc:          # a demo must never stall on the brief: fall back and say so
            if log:
                log(f"[brief] model brief failed ({str(exc)[:120]}); using the offline brief")
            b = offline_brief(episodes, title)
            b.source = f"title + screenplay statistics (model brief failed: {str(exc)[:80]})"
            return b
    return offline_brief(episodes, title)
