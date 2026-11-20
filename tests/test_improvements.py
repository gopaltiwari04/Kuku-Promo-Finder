"""Regression tests for the review fixes: learned show profile and show brief, endings that resolve (line, shot,
sound), title reveals cut at the peak, the posting lists (no repeats, score order, story-sequence cap, the premise
slot, no finale in cold lists), copy cleanup, start lead-ins, the stage-2 re-ranker (line checks, fallback, cache),
and a synthetic show in another genre.
Run:  python -m pytest -q"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.candidate_generator import CandidateGenerator  # noqa: E402
from src.evaluator import PromoEvaluator, clean_model_copy, drop_stock_clauses, strip_invented_questions  # noqa: E402
from src.llm_adapter import LLMAdapter, LLMResponse  # noqa: E402
from src.parser import EpisodeIndex, load_corpus, parse_text  # noqa: E402
from src.ranker import assign_waves, clash, rank  # noqa: E402
from src.schema import Promo  # noqa: E402
from src.scoring import (BlindContext, build_segment, cliff_profile, hook_profile, spoiler_score, title_promise,  # noqa: E402
                         title_reveal_nouns, title_tokens)
from src.parser import ts_to_sec  # noqa: E402
from src.show_profile import profile_for, story_sequences  # noqa: E402

HAVE_CORPUS = config.DEFAULT_INPUT_DIR is not None


# ------------------------------------------------------------------------------------------- show profile
def test_new_show_with_different_names_still_builds_cross_episode_promos():
    """The interview runs a NEW show: nothing about this show's cast may be hard-coded."""
    if not HAVE_CORPUS:
        return
    eps0, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    prof0 = profile_for(eps0, config.show_title_from_dir(config.DEFAULT_INPUT_DIR))
    lead = sorted(prof0.protagonists)[0].split()[-1].title()                     # learned, not typed in
    other = sorted(prof0.premise_cast - prof0.protagonists)[0].split()[-1].title()
    ren = {lead: "Marisol", other: "Viktor"}
    tmp = Path(tempfile.mkdtemp()) / "My Contract Husband. He was a Count"
    tmp.mkdir()
    try:
        for f in config.DEFAULT_INPUT_DIR.iterdir():          # read-only: originals are only read, copies renamed
            if f.is_file():
                txt = f.read_text(encoding="utf-8")
                for a, b in ren.items():
                    txt = re.sub(r"\b" + a + r"\b", b, txt)
                    txt = re.sub(r"\b" + a.upper() + r"\b", b.upper(), txt)
                (tmp / f.name).write_text(txt, encoding="utf-8")
        eps, _ = load_corpus(tmp)
        title = config.show_title_from_dir(tmp)
        prof = profile_for(eps, title)
        assert any("MARISOL" in x for x in prof.protagonists) and any("VIKTOR" in x for x in prof.premise_cast)
        assert "none" in prof.sources["aliases"]                 # no alias file for this "show"
        gen = CandidateGenerator(eps, title, profile=prof)
        cands = gen.generate_all()
        assert sum(c.promo_type == "multi_segment_cross" for c in cands) > 0
        assert len(gen.villain_swaps) > 0                        # false-payback guard still active without aliases
    finally:
        shutil.rmtree(tmp.parent)


def test_code_has_no_hard_coded_cast_or_title():
    """Nothing about the input pack's show is typed into the code: its cast, title and title noun are learned."""
    if not HAVE_CORPUS:
        return
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    title = config.show_title_from_dir(config.DEFAULT_INPUT_DIR)
    prof = profile_for(eps, title)
    names = {w for x in prof.core_cast | prof.protagonists for w in x.split() if len(w) >= 4} - {"LADY", "LORD"}
    code = "\n".join(f.read_text(encoding="utf-8") for f in list((ROOT / "src").glob("*.py")) + [ROOT / "app.py"])
    hits = sorted(n for n in names if re.search(r"\b" + n + r"\b", code, re.I))
    assert not hits, f"the code names this show's cast: {hits}"
    assert title.lower() not in code.lower()


# ------------------------------------------------------------------------------------------- cuts
def test_closing_line_is_not_a_cliffhanger():
    if not HAVE_CORPUS:
        return
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    ix = EpisodeIndex([e for e in eps if e.episode_number == 2][0])
    closing = cliff_profile(build_segment(ix, 53.5, 107.4))      # ends on a closing line
    open_cut = cliff_profile(build_segment(ix, 53.5, 76.4))      # ends on a one-word question
    assert closing.closure >= 2 and open_cut.closure == 0
    assert open_cut.score > closing.score + 2


def test_title_promise_is_a_premise_reveal_not_a_spoiler():
    toks = title_tokens("I Married a Poor Driver. He Was a Billionaire")
    ep = parse_text("""EPISODE NUMBER: Episode 45
--- SCRIPT START ---
SCENE 01
SCENE HEADING: INT. HALL - DAY
SCENE START: 00:00.00
SCENE END: 00:10.00
ACTIONS AND DIALOGUE:
**MIRA (Shocked):**
Start: (00:00.00) | End: (00:02.00)
"You are the Billionaire?"
**TOM (Calm):**
Start: (00:02.10) | End: (00:05.00)
"The rightful Billionaire of Westbrook."
--- SCRIPT END ---
""", "episode-045-x.txt")
    ix = EpisodeIndex(ep)
    ctx = BlindContext([build_segment(ix, 0.0, 5.0)])
    assert title_promise(ctx, toks) == 1
    _, twists = spoiler_score([45], ctx.all_text(), toks)
    assert twists == []                                          # the title's own twist word = its promise


# ------------------------------------------------------------------------------------------- posting lists
def _promo(pid, ep, s, e, comp, clar=8.0, hook=8.0, ptype="continuous", spoiler=1.0, seq="S1", **sig):
    seg = {"episode": ep, "start_timestamp": f"{int(s // 60):02d}:{s % 60:05.2f}",
           "end_timestamp": f"{int(e // 60):02d}:{e % 60:05.2f}", "duration_sec": round(e - s, 2)}
    p = Promo(promo_id=pid, promo_type=ptype, segments=[seg], total_duration_sec=round(e - s, 2), hook="[x]",
              cliffhanger="[x]", why_it_works="x", episodes=[ep],
              scores={"hook_strength": hook, "cliffhanger_tension": 7.0, "standalone_clarity": clar,
                      "pacing_density": 7.0, "trope_virality": 7.0, "spoiler_risk": spoiler},
              signals=dict({"hook_pass": True, "sequences": [seq]}, **sig))
    p.composite_score = comp
    return p


def test_posting_lists_hold_failures_and_never_repeat_a_scene():
    rules = dict(config.POSTING_RULES)
    a = _promo("A", 1, 10, 50, 90)
    b = _promo("B", 1, 20, 55, 89)                                     # shares footage with A -> alternate
    c = _promo("C", 1, 52, 80, 88)                                     # starts 2 s after A's cut -> alternate
    d = _promo("D", 2, 0, 30, 87)                                      # 2nd post from sequence S1 in Post first
    e = _promo("E", 3, 0, 30, 86)                                      # 3rd from S1 -> not in Post first
    f = _promo("F", 4, 0, 30, 70, seq="S2")
    prem = _promo("P", 38, 10, 50, 60, spoiler=7.0, seq="S9", premise_reveal=True)   # title reveal from Ep 38
    late = _promo("L", 40, 0, 30, 85, spoiler=6.0, seq="S8")           # spoiler 6 -> Later
    fin = _promo("Z", 50, 0, 30, 99, spoiler=1.0, seq="S7")            # finale (last 10% of 50 eps) -> never cold
    weak = _promo("W", 2, 40, 70, 95, hook=5.0, seq="S3")              # weak hook -> held, despite the best score
    res = _promo("R", 5, 0, 30, 94, seq="S4", resolved=True, resolution_reasons=["final shot resolves"])
    filler = [_promo(f"X{i}", 20 + i, 0, 30, 50.5 + i, seq=f"T{i}") for i in range(3)]
    promos = [a, b, c, d, e, f, prem, late, weak, res, fin] + filler
    info = assign_waves(promos, rules)
    assert weak.wave == 4 and weak.wave_reason.startswith("held: weak hook")
    assert res.wave == 4 and "resolves" in res.wave_reason
    assert b.variant_of == "A" and c.variant_of == "A" and "starts where A ends" in c.wave_reason
    assert a.wave == 1 and d.wave == 1 and e.wave == 2 and "holds this scene twice" in e.wave_reason
    assert prem.wave == 1 and info["premise_picks"] == 1 and info["premise_pick"] == "P"   # the reserved slot
    assert late.wave == 3
    assert info["finale_first_episode"] == 46 and fin.wave == 3 and fin.signals["audience"] == "retarget"
    assert all(max(p.episodes) < 46 for p in promos if p.wave in (1, 2))
    plan = [p for p in promos if p.wave in (1, 2, 3)]
    assert not any(clash(x, y, rules["chain_gap_sec"]) for i, x in enumerate(plan) for y in plan[i + 1:])


def test_rank_orders_each_list_by_score():
    promos = [_promo(f"C{i}", i, 0, 30, 0, hook=6.0 + 0.3 * i, seq=f"S{i}") for i in range(1, 13)]
    ranked = rank(promos)
    for w in (1, 2, 3):
        grp = [p for p in ranked if p.wave == w]
        assert grp == sorted(grp, key=lambda p: -p.composite_score)
    assert sum(p.wave == 1 for p in ranked) <= config.POSTING_RULES["post_first"]


# ------------------------------------------------------------------------------------------- endings + hooks
def _seg(ep, a, b):
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    ix = EpisodeIndex([e for e in eps if e.episode_number == ep][0])
    return build_segment(ix, ts_to_sec(a), ts_to_sec(b))


def test_endings_that_resolve_are_caught_and_cliffhangers_are_not():
    if not HAVE_CORPUS:
        return
    resolved = [(4, "02:07.40", "02:29.00"),    # a walk into the light, music swells
                (10, "00:33.20", "00:57.60"),   # a teasing line and a soft smile
                (7, "01:23.50", "01:47.80"),    # a calm walk away
                (45, "00:10.20", "00:36.80")]   # a tender moment after the reveal
    open_ = [(3, "00:25.90", "01:03.50"),       # a plea while she is dragged off
             (12, "00:35.40", "01:09.20"),      # an ultimatum
             (28, "00:47.40", "01:22.30")]      # cut to black on her face
    for ep, a, b in resolved:
        assert cliff_profile(_seg(ep, a, b)).resolved, (ep, a, b)
    for ep, a, b in open_:
        assert not cliff_profile(_seg(ep, a, b)).resolved, (ep, a, b)


def test_visual_only_opening_needs_an_action():
    if not HAVE_CORPUS:
        return
    hp = hook_profile(_seg(10, "00:33.20", "00:57.60"))      # a still picture on screen, no line until 4.9 s
    assert not hp.passes_gate and any("picture, not an action" in r for r in hp.reasons)
    assert hook_profile(_seg(3, "00:25.90", "01:03.50")).passes_gate


def test_title_reveal_is_cut_at_its_peak():
    if not HAVE_CORPUS:
        return
    title = config.show_title_from_dir(config.DEFAULT_INPUT_DIR)
    nouns = title_reveal_nouns(title)
    assert len(nouns) == 1
    assert title_reveal_nouns("My Secret Husband Is the CEO") == {"ceo"}
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    gen = CandidateGenerator([e for e in eps if e.episode_number == 45], title)
    cuts = gen.premise_cuts(45)
    assert cuts and all(20.0 <= c.total_duration_sec <= 40.0 for c in cuts)
    ix = gen.index[45]
    reveal = [d for d in ix.dialogue if re.search(r"\b" + next(iter(nouns)) + r"\b", d.text, re.I)]
    assert any(abs(c.segments[0].end_sec - reveal[0].end_sec) <= 4.5 or
               abs(c.segments[0].end_sec - reveal[-1].end_sec) <= 4.5 for c in cuts)


def test_story_sequences_group_the_wedding():
    if not HAVE_CORPUS:
        return
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    seq = story_sequences(eps)
    first = {seq[(e.episode_number, e.scenes[0].scene_id)] for e in eps if e.episode_number <= 5}
    assert len(first) == 1                                    # the Ep 1-5 wedding is one sequence
    ep6 = [e for e in eps if e.episode_number == 6][0]
    assert seq[(6, ep6.scenes[0].scene_id)] not in first


# ------------------------------------------------------------------------------------------- copy + cache
def test_model_copy_cleanup():
    out = clean_model_copy("S1 +00.00s: Speaker A sneers at SPEAKER B's gown in S2", {"A": "Vera", "B": "Maya"})
    assert out == "Vera sneers at Maya's gown in segment 2"
    assert clean_model_copy("Ep2 00:53.50 (+00.00s) - A smug rival sneers", {}) == "A smug rival sneers"
    assert strip_invented_questions("He lifts the firewood with ease; how will they survive?.") == \
        "He lifts the firewood with ease"
    assert strip_invented_questions("She whispers 'You don't even know me'. Will she retaliate?.") == \
        "She whispers 'You don't even know me'."
    assert drop_stock_clauses("Ethan coldly announces it at the altar, stopping scrollers instantly") == \
        "Ethan coldly announces it at the altar"


MINI = (ROOT / "tests" / "test_core.py").read_text(encoding="utf-8").split('MINI = """', 1)[1].split('"""', 1)[0]


class RerankAdapter(LLMAdapter):
    """A fake model for stage 2: scores clips in reverse order, writes one good line, one with a made-up quote,
    and skips one clip."""

    def __init__(self):
        os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or "test-key"
        super().__init__("gemini", "rerank-model", pacing_sec=0)
        self.requested = []

    def complete_json(self, system, user, max_tokens=2048, tag=""):
        ids = re.findall(r"### CLIP (C\d\d)", user)
        self.requested.append(ids)
        clips = [{"id": cid, "score": 10 * (k + 1), "why": "SPEAKER A is cornered, and the clip cuts before the answer.",
                  "sells_premise": False} for k, cid in enumerate(ids)]
        if len(clips) > 1:
            clips[1]["why"] = 'SPEAKER A screams "the treasure is buried under the lighthouse" and storms out.'
        if len(clips) > 2:
            clips.pop(2)
        return LLMResponse("{}", {"clips": clips}, 100, 50, 0, 0.01, "gemini", "rerank-model")


def _stage1(n_eps=3):
    from src.ranker import shortlist
    from src.show_brief import offline_brief
    eps = [parse_text(MINI.replace("Episode 7", f"Episode {n}"), f"episode-00{n}-x.txt") for n in range(1, n_eps + 1)]
    brief = offline_brief(eps, "Test Show")
    gen = CandidateGenerator(eps, "Test Show", brief=brief)
    cands = gen.generate_all()
    gen.tighten(cands)
    ev = PromoEvaluator(gen, "Test Show", brief=brief)
    promos = ev.evaluate(cands)
    shortlist(promos, size=1000)            # sets the stage-1 code score on every promo
    return gen, ev, brief, promos


def test_model_reranker_orders_checks_lines_and_falls_back_per_clip():
    from src.reranker import ModelReranker
    cache = Path(tempfile.mkdtemp())
    orig = config.LLM_CACHE_DIR
    config.LLM_CACHE_DIR = cache
    try:
        gen, ev, brief, promos = _stage1()
        assert len(promos) >= 3
        ad = RerankAdapter()
        rr = ModelReranker(ad, ev, brief, group_size=10)
        out = rr.rerank(promos)
        assert set(out) == {p.promo_id for p in promos}
        assert all("Secret logline" not in "".join(ids) for ids in ad.requested)
        by = {r.by for r in out.values()}
        assert by == {"gemini:rerank-model", "keyword"}                        # one skipped clip kept its code score
        assert rr.stats["lines_rejected"] >= 1 and rr.stats["lines_kept"] >= 1   # the made-up quote was refused
        assert not any("lighthouse" in r.why for r in out.values())
        kept = [r for r in out.values() if r.by != "keyword" and "cornered" in r.why]
        assert kept and all("SPEAKER" not in r.why for r in kept)              # tags turned back into names
        # second run: every group comes from the cache, no model call
        ad2 = RerankAdapter()
        rr2 = ModelReranker(ad2, ev, brief, group_size=10)
        rr2.rerank(promos)
        assert ad2.requested == [] and rr2.stats["cached_groups"] == rr2.stats["groups"]
    finally:
        config.LLM_CACHE_DIR = orig
        shutil.rmtree(cache)


def test_reranker_error_or_time_budget_keeps_keyword_order():
    from src.llm_adapter import LLMError
    from src.reranker import ModelReranker

    class Broken(RerankAdapter):
        def complete_json(self, system, user, max_tokens=2048, tag=""):
            raise LLMError("503 from the provider")
    gen, ev, brief, promos = _stage1()
    rr = ModelReranker(Broken(), ev, brief, use_cache=False, group_size=2)
    out = rr.rerank(promos)
    assert all(r.by == "keyword" for r in out.values()) and rr.stats["fallback_groups"] == rr.stats["groups"]
    assert all(out[p.promo_id].score == p.signals["code_score"] for p in promos)
    rr = ModelReranker(RerankAdapter(), ev, brief, use_cache=False, group_size=2, time_budget=-1)
    assert all(r.by == "keyword" for r in rr.rerank(promos).values())


def test_check_line_refuses_invented_quotes_questions_and_stock_phrases():
    from src.reranker import check_line
    spoken = ["You stole my wedding!", "Who killed him?"]
    assert check_line("Maya is accused of stealing a wedding, and it cuts on \u201cWho killed him?\u201d", spoken)[0]
    assert not check_line("Maya says \u201cI never loved you\u201d and walks out of the church.", spoken)[0]
    assert not check_line("Who will save Maya now that her wedding is ruined?", spoken)[0]
    assert not check_line("short", spoken)[0]


def test_starts_get_a_short_lead_in_before_the_first_line():
    gen, ev, brief, promos = _stage1()
    moved = [p for p in promos if p.signals.get("tightened")]
    assert moved
    for p in moved:
        for s in p.segments:
            if s.get("pre_roll_sec"):
                gap = ts_to_sec(s["anchor_timestamp"]) - ts_to_sec(s["start_timestamp"])
                assert 0 < gap <= config.PRE_ROLL_SEC + 1e-6 and abs(gap - s["pre_roll_sec"]) < 0.011
        assert config.MIN_DURATION - 1e-6 <= p.total_duration_sec <= config.MAX_DURATION + 1e-6


def test_a_closing_ending_is_held_before_any_model_sees_it():
    if not HAVE_CORPUS:
        return
    from src.ranker import gate, shortlist
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    gen = CandidateGenerator([e for e in eps if e.episode_number == 2], "x")
    from src.schema import Candidate
    cand = Candidate(candidate_id="T", promo_type="continuous", segments=[gen._seg(gen.index[2], 53.5, 107.4, "main")])
    p = PromoEvaluator(gen, "x").evaluate([cand])[0]
    p.scores["hook_strength"] = 9.0                       # even with a perfect hook ...
    assert p.signals.get("resolved") and "resolves" in (gate(p, config.POSTING_RULES) or "")
    assert shortlist([p]) == []                           # ... it never reaches the re-ranker


def test_lexicons_hold_no_show_specific_words():
    """The words this show's script makes high-stakes (learned per show) must not be typed into the scoring code."""
    if not HAVE_CORPUS:
        return
    from src import lexicons, scoring
    from src.show_brief import offline_brief
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    b = offline_brief(eps, config.show_title_from_dir(config.DEFAULT_INPUT_DIR))
    banned = set(b.stakes_terms) | set(b.twist_terms)
    assert banned
    for mod in (lexicons, scoring):
        src = Path(mod.__file__).read_text(encoding="utf-8").lower()
        code = "\n".join(ln.split("#")[0] for ln in src.splitlines())
        # a learned word typed in as its own entry ("poison", |poison|); generic phrases like "destroy you" are fine
        hits = sorted(w for w in banned if re.search(r"[\"'|(:]" + re.escape(w) + r"\w{0,3}[\"'|)?]", code))
        assert not hits, f"{mod.__name__} hard-codes {hits}"


def test_show_brief_learns_stakes_words_from_the_script():
    if not HAVE_CORPUS:
        return
    from src.show_brief import offline_brief
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    b = offline_brief(eps, config.show_title_from_dir(config.DEFAULT_INPUT_DIR))
    assert b.premise_twist and b.twist_terms and len(b.stakes_terms) >= 3
    cast = {w.lower() for e in eps for d in e.dialogue for w in re.findall(r"[A-Za-z]+", d.speaker_raw)}
    assert not set(b.stakes_terms) & cast                                      # names are never stakes words


def test_adapter_cache_replays_without_network():
    cache = Path(tempfile.mkdtemp())
    orig = config.LLM_CACHE_DIR
    config.LLM_CACHE_DIR = cache
    try:
        ad = RerankAdapter()
        data, hit = ad.complete_json_cached("sys", "### CLIP C01\nx", 100)
        assert not hit and data["clips"][0]["id"] == "C01"
        data2, hit2 = RerankAdapter().complete_json_cached("sys", "### CLIP C01\nx", 100)
        assert hit2 and data2 == data
        ro = LLMAdapter("gemini", "rerank-model", api_key="k", cache_only=True)
        assert ro.complete_json_cached("sys", "### CLIP C01\nx", 100)[1]           # cache-only replays it
        try:
            ro.complete_json_cached("sys", "something new", 100)
            raise AssertionError("cache-only must not call the network")
        except Exception as exc:
            assert "cache-only" in str(exc)
    finally:
        config.LLM_CACHE_DIR = orig
        shutil.rmtree(cache)


def test_solution_doc_matches_the_outputs():
    """The 2-page doc is generated from raw_outputs/; a stale doc (numbers that no longer match a run) fails here."""
    out, doc = config.DEFAULT_OUTPUT_DIR, config.PROJECT_ROOT / "docs" / "SOLUTION_DOC.md"
    if not (doc.exists() and (out / "promos_all_episodes.json").exists()):
        return                                    # a public clone has neither (they quote the screenplay)
    if not ((out / "promos_all_episodes.json").exists() and doc.exists()):
        return
    sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
    import build_doc
    D = build_doc.load()
    md = build_doc.to_md(build_doc.blocks(D))
    assert md == doc.read_text(encoding="utf-8"), "run: python scripts/build_doc.py"
    first = min((p for p in D["promos"] if p.get("wave") == 1), key=lambda p: p["post_order"])
    s = first["segments"][0]
    assert f"Ep{s['episode']} {s['start_timestamp']}" in md
    assert "?" not in re.findall(r"(\S+) promos would", md)


# ------------------------------------------------------------------------------------------- a show in another genre
def test_a_new_genre_show_runs_end_to_end():
    """3 synthetic episodes of a modern CEO drama in a slightly different format (hyphenated headings, (O.S.)
    speakers, a stranger who is named later): every hard check passes, the lead and the title reveal are found,
    and the clip that ends with the couple walking off into the night is held."""
    from src.pipeline import run_folder
    d = ROOT / "tests" / "fixtures" / "My Secret Husband Is the CEO"
    out = Path(tempfile.mkdtemp())
    try:
        res = run_folder(d, out)
        assert res.verification["all_pass"], {k: v for k, v in res.verification["checks"].items() if not v["pass"]}
        assert res.meta["show_profile"]["protagonists"] == ["MIA"]
        assert any(p.signals.get("premise_reveal") for p in res.promos)               # "You're the CEO?"
        walk_off = [p for p in res.promos if p.signals.get("resolved")]
        assert walk_off and all(p.wave == 4 for p in walk_off)
        assert all(20.0 <= p.total_duration_sec <= 60.0 for p in res.promos)
        # saved outputs keep only the hook and cut lines: no hook-window transcripts, no full prompts
        import json
        saved = json.loads((out / "promos_all_episodes.json").read_text(encoding="utf-8"))
        assert all("hook_window_lines" not in p["signals"] for p in saved["promos"])
        meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
        assert all("user" not in s for s in meta.get("sample_llm_prompts", []))
        assert meta["story_sequences"]
        # same input, same plan: the keyword scorer is deterministic
        out2 = Path(tempfile.mkdtemp())
        again = run_folder(d, out2)
        shutil.rmtree(out2)
        plan = lambda r: [(p.promo_id, p.wave, p.post_order) for p in sorted(r.promos, key=lambda p: p.post_order)]
        assert plan(res) == plan(again)
    finally:
        shutil.rmtree(out)


# ------------------------------------------------------------------------------------------- copy names the right act
def test_copy_describes_the_line_it_quotes():
    from src.evaluator import line_type
    from src.lexicons import dialogue_acts, dialogue_arousal
    assert line_type("You framed me!") == "accusation"
    assert line_type("Guards! Remove him at once!") == "command"
    assert line_type("I did not steal it! Look at her eyes!") == "denial"
    assert "threat" not in dialogue_acts("What if you never come back?")             # a fear, not a threat
    assert "accusation" not in dialogue_acts("You cannot promise that.")             # not shouted, not a charge
    assert "threat" not in dialogue_acts("They see your men outside the back door.")
    assert dialogue_arousal("Ethan!", ["sobbing", "stunned"]) <= 1.5               # a name alone is not a hook
    assert dialogue_arousal("Good.", ["cold"]) <= 1.5


# ------------------------------------------------------------------------------------------- second judge maths
def test_second_judge_metrics_and_repetition():
    from src.quality_eval import judge_metrics, repetition_of
    rows = [{"hook": "S", "ending": "O", "makes_sense_alone": "Y", "would_post": "Y"},
            {"hook": "W", "ending": "R", "makes_sense_alone": "P", "would_post": "N"},
            {"hook": "S", "ending": "O", "makes_sense_alone": "P", "would_post": "M"}]
    m = judge_metrics(rows)
    assert m["post_score"] == 0.5 and m["all_three_tests"] == 1
    assert m["weak_or_closed_endings"] == 1 and m["weak_hooks"] == 1 and m["would_not_post"] == 1
    spans = {"S001": [[1, 0.0, 100.0], [2, 0.0, 50.0]], "S002": [[3, 0.0, 90.0]]}
    r = repetition_of(["Ep1 00:10.00-00:40.00", "Ep2 00:05.00-00:30.00", "Ep3 00:01.00-00:30.00 + Ep2 00:31.00-00:45.00"],
                      spans)
    assert r["most_from_one_sequence"] == 3 and r["that_sequence"] == "S001" and r["episodes"] == 3


def test_blind_eval_merge_scores_sources_win_rate_and_agreement():
    import csv
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("blind_eval", ROOT / "scripts" / "blind_eval.py")
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    tmp = Path(tempfile.mkdtemp())
    try:
        be.EVAL = tmp
        ids = [f"C{i:02d}" for i in range(1, 11)]
        key = {c: {"source": "pipeline" if i % 2 else "baseline", "promo_id": c, "segments": []} for i, c in enumerate(ids)}
        (tmp / "key.json").write_text(json.dumps({"clips": key}))
        good = lambda c: {"id": c, "hook": 5, "ending": 4, "clarity": 4, "would_post": "Y", "note": ""}
        bad = lambda c: {"id": c, "hook": 2, "ending": 2, "clarity": 3, "would_post": "N", "note": ""}
        rows = [good(c) if key[c]["source"] == "pipeline" else bad(c) for c in ids]
        (tmp / "judge1_fake.json").write_text(json.dumps({"judge": "gemini:fake", "clips": rows}))
        rows2 = [dict(r, hook=r["hook"] - 1 if r["hook"] > 1 else 1) for r in rows]
        (tmp / "judge2.json").write_text(json.dumps({"judge": "other-model", "clips": rows2}))
        with open(tmp / "manual.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "hook", "ending", "clarity", "would_post", "note"])
            for c in ids:
                w.writerow([c, 3, 3, 3, "m", ""]) if c != "C03" else w.writerow([c, "", "", "", "", ""])
        rep = be.merge(None, tmp / "judge2.json", tmp / "manual.csv")
        j1 = rep["judges"]["Judge 1 (gemini:fake)"]
        assert j1["pipeline"]["n"] == 5 and j1["baseline"]["n"] == 5
        assert j1["delta"]["hook"] == 3.0 and j1["delta"]["would_post"] == 1.0 and j1["win_rate"] == 1.0
        person = rep["judges"]["Person (manual labels)"]
        assert person["clips_scored"] == 9 and person["win_rate"] == 0.5          # a blank row is skipped
        assert any(a["pair"].startswith("Judge 1") and a["same_would_post_pct"] == 100.0 for a in rep["agreement"])
        assert rep["pooled"]["delta"]["hook"] > 0 and (tmp / "results.md").exists()
    finally:
        shutil.rmtree(tmp)

