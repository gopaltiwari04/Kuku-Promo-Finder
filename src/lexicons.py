"""Deterministic drama lexicons + per-block feature extractors.

These are the cheap, transparent signals the candidate generator uses to shortlist windows and the keyword scorer
uses to rate them. Everything here is genre-general (family, marriage, money, crime, threats, pleas). Nothing about a
particular show is hard-coded: the words that matter in THIS show (its premise nouns and high-stakes vocabulary) are
learned per run by src/show_brief.py and installed with set_show_terms().
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _rx(words: Iterable[str]) -> re.Pattern:
    alts = sorted({w.strip() for w in words if w.strip()}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.I)


# ---- 1. Emotion / delivery tags (from "**SPEAKER (Cold, Commanding):**") ------------------
EMOTION_WEIGHTS: Dict[str, float] = {}
for w in ("screaming furious outraged accusing cruel venomous menacing threatening hysterical vicious "
          "sadistic enraged roaring gloating mocking taunting vindictive malicious deadly dangerous "
          "vengeful brutal unhinged savage sneering spiteful desperate begging agonized agonised panicked "
          "horrified heartbroken devastated betrayed piercing imperious scornful derisive contemptuous "
          "condescending callous commanding shouting booming frantic terrified screeching murderous "
          "seething livid wrathful merciless heartless sadistically shrieking wailing sobbing crying "
          "anguished tormented ruthless icy").split():
    EMOTION_WEIGHTS[w] = 3.0
for w in ("defiant cold smug shocked harsh disgusted dismissive arrogant fierce triumphant incredulous "
          "pleading bitter mean sarcastic stern accusatory horrified indignant hurt raw outraged "
          "authoritative grim severe firm-voiced resolute unforgiving blunt haughty proud hostile "
          "suspicious alarmed urgent stunned disbelieving gasping choking breaking strangled weeping "
          "trembling furiously").split():
    EMOTION_WEIGHTS.setdefault(w, 2.0)
for w in ("firm choked breathless confused shaken tense nervous anxious worried uneasy serious "
          "determined steady low intense pained vulnerable exhausted weak hoarse whispering urgent").split():
    EMOTION_WEIGHTS.setdefault(w, 1.0)
LOW_AROUSAL_TAGS = set(("soft gentle tender warm loving reassuring calm quiet reflective content playful "
                        "amused relieved grateful peaceful affectionate sincere earnest devoted hesitant "
                        "curious thoughtful soothing kind fond smiling wistful nostalgic shy cheerful "
                        "comforting encouraging softly gently warmly").split())


def emotion_arousal(tags: List[str]) -> float:
    """0..3 arousal from the parenthetical delivery tags; tender tags pull it down."""
    best, low = 0.0, 0
    for tag in tags:
        for tok in re.findall(r"[a-z\-]+", tag.lower()):
            best = max(best, EMOTION_WEIGHTS.get(tok, 0.0))
            if tok in LOW_AROUSAL_TAGS:
                low += 1
    if best == 0.0 and low:
        return -1.0
    return best - 0.5 * min(low, 2) if best else 0.0


# ---- 2. Spoken anchors: instant cold-viewer comprehension -------------------------------
RELATIONAL_ANCHORS = [
    "husband", "wife", "sister", "brother", "father", "mother", "stepmother", "daughter", "son",
    "marry", "marrying", "married", "marriage", "wedding", "bride", "groom", "fianc[eé]e?",
    "family", "steal", "stole", "stolen", "betray\\w*", "cheat\\w*", "affair",
    "child", "heir", "pregnan\\w*", "doctor", "contract", "divorce", "prison", "police", "kill\\w*",
    "debt", "forged", "lie", "lied", "liar", "arrest\\w*", "thief", "theft", "trial", "inheritance",
    "disown\\w*", "mistress", "widow", "murder\\w*", "orphan", "baby", "nobody",
]
ANCHOR_RX = re.compile(r"\b(?:" + "|".join(RELATIONAL_ANCHORS) + r")\b", re.I)

LORE_PATTERNS = [
    r"\bthat night\b", r"\bwhat happened\b", r"\bafter everything\b", r"\b(?:he|she|you) promised\b",
    r"\bremember when\b", r"\bback then\b", r"\blast time\b", r"\bthe plan\b", r"\bour (?:deal|plan|arrangement)\b",
    r"\bthe (?:ring|letter|journal|book|documents?|file|photo|recording|necklace)\b", r"\bagain\b",
    r"\bhe knows\b", r"\bshe knows\b", r"\bthey know\b", r"\bit was (?:him|her|them)\b",
]
LORE_RX = re.compile("|".join(LORE_PATTERNS), re.I)
PRONOUN_RX = re.compile(r"\b(?:he|she|him|her|his|they|them|their|it)\b", re.I)

# ---- 3. Dialogue acts -------------------------------------------------------------------
QUESTION_RX = re.compile(r"\?\s*[\"”']?\s*$|\?!|!\?")
THREAT_RX = _rx([
    "will die", "will pay", "will regret", "will answer", "kill you", "kill her", "kill him", "destroy you",
    "ruin you", "arrest", "remove her", "remove him", "take her", "lock her", "you will", "or watch", "or else",
    "you're finished", "you are finished", "i will end", "dead by", "you will never see", "you'll never see",
    "pay for this", "you'll pay", "you'll regret", "you'll die", "you will die", "i will destroy", "i'll destroy",
])
REVEAL_RX = _rx([
    "is marrying me", "he is marrying", "i am the", "you are the", "is my son", "my son", "the rightful",
    "alive", "died", "is dead", "the truth", "i knew", "you knew", "of course i knew",
    "never going to marry", "true identity", "who you really are", "the real",
    "your husband", "his mother", "imposter", "framed", "she is innocent", "it was you", "you did this",
    "pregnant", "heir", "was never yours", "planted", "confess",
])
# A line that merely starts with "You" is an accusation only when it is shouted ("You lied to me!");
# "You cannot promise that." is not.
ACCUSE_RX = re.compile(
    r"^\s*(?:you\b(?!'re welcome)[^.?]*!|how (?:dare|could)|liar\b|murder\w*|thief\b|traitor\b)|\byou (?:knew|let|"
    r"stole|lied|swore|did this|planted|framed|betrayed|abandoned|killed|cheated)\b", re.I)
# an order, optionally after a vocative ("Guards, take him away!", "Mia, get out")
COMMAND_RX = re.compile(
    r"^\s*(?:[A-Za-z]+[,!]\s+)?(?:remove|leave\b(?! where)|get out|stop|look for|look at me|take|turn around|kneel|"
    r"arrest|let (?:her|him|me) go|throw|bring|stand aside|step aside|enough|silence|out\b|go\b|run\b|drop|"
    r"release|answer me|say it|tell (?:them|me|him|her)|don't touch|take your hands|hold (?:her|him)|move\b|lock|"
    r"fetch|call|sign)", re.I)
VOW_RX = _rx(["i will", "i swear", "i'll make", "never again", "we settle", "you will pay", "i promise you",
              "mark my words", "i will not", "i won't", "they will never", "no one will", "this is not over",
              "i will find", "i will take back", "i will reclaim"])
PLEA_RX = _rx(["please", "i beg", "i'm begging", "help me", "someone", "don't do this", "don't leave",
               "save him", "save her", "let him go", "have mercy", "mercy"])
OPENER_DEPENDENT_RX = re.compile(r"^\s*(?:\.\.\.|…|because\b|and\b|but\b|so\b|yes\b|no\b(?!\.\.\.)|then\b|"
                                 r"well\b|that\b|which\b|or\b|also\b|too\b)", re.I)
CONTINUES_RX = re.compile(r"(?:\.\.\.|…|—|--)\s*[\"”']?\s*$")
STARTS_CONTINUATION_RX = re.compile(r"^\s*[\"“']?\s*(?:\.\.\.|…)")

# ---- 4. Visual / audio intensity ---------------------------------------------------------
KINETIC_STRONG = [
    "slaps?", "slapped", "strikes?", "struck", "punch\\w*", "shoves?", "shoved", "hurls?", "rips?", "ripped",
    "seizes?", "seized", "drags?", "dragged", "drops? to (?:her|his) knees", "collapses?", "collapsed",
    "crash\\w*", "smash\\w*", "shatter\\w*", "slams?", "slammed", "kicks?", "kicked", "lunges?", "lunged",
    "draws? (?:a|his|her) (?:blade|knife|pistol|gun|weapon)", "chokes?", "stabs?", "explod\\w*", "explosion",
    "whips?", "bursts? (?:open|in|through)", "storms? (?:in|into)", "charges?", "tackles?",
    "pins?", "blood\\w*", "lacerat\\w*", "flings?", "yanks?", "wrench\\w*", "throws?", "threw", "tears? (?:her|his|the)",
    "grabs?", "grabbed", "blocks? the", "headbutt\\w*", "bleeding", "ablaze", "inferno",
    "falls? to", "topples?", "sinks? to one knee", "kneels?", "backhand\\w*", "spits?",
]
KINETIC_MEDIUM = [
    "clutch\\w*", "clench\\w*", "grips?", "gripping", "trembl\\w*", "recoils?", "flinch\\w*", "staggers?",
    "stumbles?", "pushes?", "points? (?:at|toward)", "tears? (?:spill|stream)", "sobs?", "gasps?", "rushes?",
    "chains?", "manacle\\w*", "shackle\\w*", "struggl\\w*", "resist\\w*", "fist", "white-knuckled", "steps? between",
    "kiss\\w*", "embrace\\w*", "snatch\\w*", "rise[sn]?", "whirls?", "spins? around", "freezes?",
]
KINETIC_STRONG_RX = re.compile(r"\b(?:" + "|".join(KINETIC_STRONG) + r")\b", re.I)
KINETIC_MEDIUM_RX = re.compile(r"\b(?:" + "|".join(KINETIC_MEDIUM) + r")\b", re.I)
SFX_IMPACT_RX = _rx(["slap", "punch", "crash", "smash", "shatter", "explosion", "explodes", "thud", "gasp",
                     "scream", "whip crack", "crack", "gunshot", "slam", "bang", "bangs", "splinter", "roar",
                     "crunch", "impact", "collaps", "pounding", "kicked open", "clatter", "boom"])
SFX_CLIFF_RX = _rx(["cliffhanger", "ominous", "drone", "swell", "swells", "sting", "stinger", "boom", "peak",
                    "abrupt", "sudden", "silence", "creak", "footsteps", "pounding", "gasp", "hold", "tense",
                    "dramatic", "thunder"])
CLOSEUP_RX = _rx(["extreme close-up", "close-up", "push-in", "push in", "pushes in", "tight", "insert", "ecu",
                  "slow push", "macro"])
TRANSITION_RX = re.compile(r"^\s*(?:cut to black|fade to black|fade out|fades? to black|smash cut to black|"
                           r"black screen|end credits|fade to white)\b", re.I)
SILENT_VISUAL_RX = _rx(["stares", "gazes", "in silence", "silently", "looks on", "watches", "lingers", "holds the gaze",
                        "slowly", "walks", "steps into", "tracking shot", "establishing", "wide shot", "pans",
                        "drifts", "lingering", "breathes", "pauses"])
RESOLUTION_RX = _rx(["embrace", "embraces", "hug", "hugs", "smiles warmly", "laughs", "relief", "relieved",
                     "tender", "gently", "walk away together", "into the light", "falls asleep", "peaceful", "safe now",
                     "thank you", "warm", "soft smile", "rests", "smiles softly", "fading out", "fades out",
                     "together", "content"])
# An ending that RESOLVES the scene instead of leaving it hanging, read from the final shots and sound
# (a couple walking into the sunlight, a soft smile, a triumphant swell fading out). Generic, not show-specific.
RESOLVE_VISUAL_RX = re.compile("|".join([
    r"\bwalk(?:s|ing)?\b[^.]{0,60}\b(?:away|off|out|together|hand[- ]in[- ]hand|into the|through the|past|"
    r"toward (?:the|their|his|her) (?:door|home|house|car))",
    r"\binto the (?:sun ?light|daylight|light|distance|sunset|night air)\b",
    r"\b(?:doors?|gates?) (?:swing|swings|creak|creaks|open|opens)\b[^.]{0,50}\b(?:daylight|sunlight|light)\b",
    r"\bfad(?:e|es|ing) (?:out|away)\b", r"\bscreen fades\b",
    r"\btriumphant\b", r"\bvictorious\b",
    r"\b(?:embrace|embraces|embracing|hug|hugs|hugging|kiss|kisses|kissing)\b",
    r"\bforeheads? (?:pressed|touching|resting)\b",
    r"\bsmil(?:e|es|ing)\b[^.]{0,25}\b(?:softly|warmly|tenderly|gently|sweetly)\b",
    r"\b(?:soft|warm|tender|gentle|genuine|sweet) smile\b",
    r"\blaugh(?:s|ing)? (?:together|softly)\b",
    r"\b(?:tenderly|lovingly|affectionately|gently) (?:cradles|holds|strokes|caresses|touches)\b",
    r"\bholds (?:her|his) face\b", r"\bquiet affection\b", r"\bsoften(?:s|ing|ed)? with\b", r"\bempathy\b",
    r"\b(?:relief|relieved|peaceful|serene|at peace|contented)\b", r"\bcalm, dignified\b",
]), re.I)
# sound that resolves (a triumphant or gentle score fading out) is not a suspense sting
RESOLVE_SFX_RX = re.compile(r"\b(?:triumphant|uplifting|gentle|soft|warm|tender|romantic|peaceful|joyful|"
                            r"fad(?:e|es|ing) (?:out|away)|resolv\w*|sweet)\b", re.I)
# physical ACTIONS (a slap, a drag, a fall). A visual-only opening needs one of these to count as a hook:
# a wound, a weapon or a costume on screen is a picture, not an event.
KINETIC_ACTION_RX = re.compile(r"\b(?:" + "|".join([
    "slaps?", "slapped", "strikes?", "struck", "punch\\w*", "shoves?", "shoved", "hurls?", "rips?", "ripped",
    "seizes?", "seized", "drags?", "dragged", "drops? to (?:her|his) knees", "collapses?", "collapsed",
    "crash\\w*", "smash\\w*", "shatter\\w*", "slams?", "slammed", "kicks?", "kicked", "lunges?", "lunged",
    "draws? (?:a|his|her) (?:blade|knife|pistol|gun|weapon)", "chokes?", "stabs?", "explod\\w*",
    "whips?", "bursts? (?:open|in|through)", "storms? (?:in|into)", "charges?", "tackles?", "pins?", "flings?",
    "yanks?", "wrench\\w*", "throws?", "threw", "grabs?", "grabbed", "headbutt\\w*", "falls? to", "topples?",
    "backhand\\w*", "spits?", "shoots?", "fires?", "aims? (?:a|the|his|her) (?:gun|pistol)", "tears? (?:her|his|the)",
]) + r")\b", re.I)

EVOCATIVE_PROSE_RX = _rx(["bright", "joyful", "elegant", "grand", "beautiful", "radiant", "anticipation",
                          "vintage", "gown", "silk", "golden", "majestic", "sweeping", "lush", "ornate", "glowing"])

# ---- 5. Trope / arc tags -------------------------------------------------------------------
TROPE_KEYWORDS: Dict[str, List[str]] = {
    "altar_betrayal": ["altar", "wedding", "bride", "groom", "marry", "marrying", "married", "fianc\\w*", "vows",
                       "gown", "aisle", "changed my mind", "call off", "jilted"],
    "family_humiliation": ["family", "father", "mother", "sister", "stepmother", "daughter", "disown\\w*", "disgrace\\w*",
                           "embarrass\\w*", "humiliat\\w*", "worth", "inheritance", "banish\\w*", "discarded",
                           "nothing to offer", "choose", "worthless"],
    "status_reversal": ["title", "heir", "rightful", "kneel", "bow", "nobody", "poor", "rich", "owner", "beneath",
                        "who are you", "you already know"],
    "secret_alliance_or_rescue": ["take your hands off", "let her go", "don't touch her", "rescue", "save",
                                  "protect", "stand aside", "are you hurt", "stranger", "loyal", "ally", "free her",
                                  "unlock\\w*", "release her"],
    "public_confrontation": ["guests", "everyone", "crowd", "court", "trial", "witness", "hear me", "all of you",
                             "in front of", "gasp", "publicly"],
    "revenge_payback": ["settle", "pay for", "regret", "vengeance", "revenge", "repay", "take back", "reclaim",
                        "punish", "you will answer", "what you stole", "this is not over", "you will pay"],
    "framed_accused": ["theft", "thief", "arrest\\w*", "police", "guilty", "framed", "testimony", "evidence",
                       "planted", "accus\\w*", "sentence\\w*", "prison", "innocent"],
    "cruelty_torment": ["blood\\w*", "chain\\w*", "whip\\w*", "burn\\w*", "starv\\w*", "beg", "torment\\w*",
                        "lacerat\\w*", "slap\\w*", "cruel\\w*"],
    "secret_identity": ["who are you", "identity", "alive", "imposter", "impostor", "disguise\\w*",
                        "you cannot be him", "true name", "my son", "who he is", "who you really are"],
    "romance": ["kiss\\w*", "love", "loved", "my wife", "my husband", "heart", "darling", "hold me", "embrace\\w*"],
}
TROPE_RX = {k: re.compile(r"\b(?:" + "|".join(v) + r")\b", re.I) for k, v in TROPE_KEYWORDS.items()}
TROPE_WEIGHTS = {
    "altar_betrayal": 1.0, "status_reversal": 0.95, "family_humiliation": 0.9, "secret_alliance_or_rescue": 0.9,
    "revenge_payback": 0.9, "public_confrontation": 0.85, "secret_identity": 0.85, "framed_accused": 0.8,
    "cruelty_torment": 0.75, "romance": 0.55,
}
TROPE_LABELS = {
    "altar_betrayal": "Altar betrayal", "family_humiliation": "Family humiliation",
    "status_reversal": "Status reversal", "secret_alliance_or_rescue": "Protector rescue",
    "public_confrontation": "Public confrontation", "revenge_payback": "Revenge payback",
    "framed_accused": "Framed & condemned", "cruelty_torment": "Visceral cruelty",
    "secret_identity": "Hidden identity", "romance": "Scarred romance",
}
SETUP_TAGS = {"altar_betrayal", "family_humiliation", "public_confrontation", "framed_accused", "cruelty_torment"}
PAYOFF_TAGS = {"status_reversal", "revenge_payback", "secret_alliance_or_rescue", "secret_identity"}

# Reveal words that would be spoilers *unless* the show title already promises them.
# late-story reveals (spoiler cues). Generic: identity / death / secret reveals, not this show's plot.
TWIST_RX = _rx(["alive", "is dead", "imposter", "impostor", "not who", "true identity", "real name", "secretly",
                "in disguise", "you cannot be", "cannot be him", "cannot be her", "rightful", "heir",
                "pregnant", "my son", "my daughter", "your father", "your mother", "the true", "the real"])
# a scene that is closing rather than hanging: delivery, words and final shot of the last line
CLOSURE_TAG_RX = _rx(["triumphant", "resolute", "proud", "calm", "dignified", "composed", "serene", "relieved",
                      "content", "peaceful", "victorious", "satisfied", "grateful", "at peace", "tender", "warm"])
DEPARTURE_RX = _rx(["leave", "leaving", "goodbye", "farewell", "walk away", "i'm done", "i am done", "it is over",
                    "it's over", "never see", "we are done", "let us go", "let's go", "go home", "finally free",
                    "at last", "i forgive you", "thank you"])
FINAL_BEAT_RX = _rx(["final proclamation", "final beat", "final words", "head high", "head held high", "walks out",
                     "walks away", "turns and leaves", "exits", "leaves the", "closure", "triumphant", "at peace",
                     "settles", "exhales in relief"])
REVEAL_MARK_RX = _rx(["you are", "he is", "she is", "i am", "is the", "was the", "the true", "the real", "rightful",
                      "secret", "secretly", "cannot be", "alive", "revealed", "truth", "really"])


# ---- 6. Words learned from THIS show (src/show_brief.py), installed per run ------------------
_SHOW_TERMS: Dict[str, Optional[re.Pattern]] = {"stakes": None, "anchors": None, "premise": None}


def _terms_rx(words: Iterable[str]) -> Optional[re.Pattern]:
    ws = sorted({w.strip().lower() for w in words if w and w.strip()}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in ws) + r")\b", re.I) if ws else None


def set_show_terms(stakes: Iterable[str] = (), premise: Iterable[str] = ()) -> None:
    """Install the show's own vocabulary: high-stakes words (what is at risk in this story) and premise nouns (the
    title's promise). Stakes words raise a line's arousal; both count as anchors a stranger can hold on to."""
    stakes = [w for w in stakes if w]
    _SHOW_TERMS["stakes"] = _terms_rx(stakes)
    _SHOW_TERMS["anchors"] = _terms_rx(list(stakes) + [w for w in premise if w])
    _SHOW_TERMS["premise"] = _terms_rx([w for w in premise if w])


def show_terms_in(text: str, kind: str = "stakes") -> Set[str]:
    rx = _SHOW_TERMS.get(kind)
    return {m.lower() for m in rx.findall(text or "")} if rx else set()


@lru_cache(maxsize=20000)
def kinetic_score(text: str) -> float:
    """0..3 physical intensity of a visual/SFX description."""
    if not text:
        return 0.0
    s = 0.0
    n_strong = len(KINETIC_STRONG_RX.findall(text))
    n_med = len(KINETIC_MEDIUM_RX.findall(text))
    if n_strong:
        s = 2.2 + 0.4 * min(n_strong - 1, 2)
    elif n_med:
        s = 1.0 + 0.3 * min(n_med - 1, 2)
    return min(s, 3.0)


@lru_cache(maxsize=20000)
def dialogue_acts(text: str) -> Tuple[str, ...]:
    acts: List[str] = []
    if QUESTION_RX.search(text):
        acts.append("question")
    if THREAT_RX.search(text):
        acts.append("threat")
    if REVEAL_RX.search(text):
        acts.append("reveal")
    if ACCUSE_RX.search(text):
        acts.append("accusation")
    if COMMAND_RX.search(text):
        acts.append("command")
    if VOW_RX.search(text):
        acts.append("vow")
    if PLEA_RX.search(text):
        acts.append("plea")
    return tuple(acts)


ACT_WEIGHTS = {"reveal": 1.5, "threat": 1.5, "accusation": 1.2, "command": 1.0, "question": 0.7,
               "vow": 1.1, "plea": 1.0}


def dialogue_arousal(text: str, tags: List[str]) -> float:
    """0..5 arousal of one spoken line = delivery tags + dialogue acts + punctuation + anchors."""
    a = max(emotion_arousal(tags), -1.0)
    acts = dialogue_acts(text)
    if acts:
        a += max(ACT_WEIGHTS[x] for x in acts) + 0.25 * (len(acts) - 1)
    if "!" in text:
        a += 0.6
    n_anchor = len(set(m.lower() for m in ANCHOR_RX.findall(text)))
    a += min(0.35 * n_anchor, 0.9)
    a += min(0.5 * len(show_terms_in(text, "stakes")), 1.0)      # this show's own high-stakes words
    if len(text.split()) <= 6 and ("!" in text or "?" in text):
        a += 0.3
    if not acts and len(re.findall(r"[A-Za-z']+", text)) <= 2:
        a = min(a, 1.5)          # a name or a one-word reply ("Ethan!", "Good.") is a reaction, not a hook
    return max(0.0, min(a, 5.0))


def anchors_in(text: str) -> Set[str]:
    return {m.lower() for m in ANCHOR_RX.findall(text or "")} | show_terms_in(text, "anchors")


def trope_hits(text: str) -> Dict[str, int]:
    return {k: len(rx.findall(text)) for k, rx in TROPE_RX.items() if rx.search(text)}
