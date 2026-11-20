"""Core regression tests. Run:  python -m pytest -q   (or)   python tests/test_core.py"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.candidate_generator import CandidateGenerator  # noqa: E402
from src.evaluator import PromoEvaluator  # noqa: E402
from src.llm_adapter import LLMAdapter  # noqa: E402
from src.parser import EpisodeIndex, load_corpus, parse_text, slice_window, ts_to_sec  # noqa: E402
from src.ranker import rank  # noqa: E402

MINI = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: Secret logline naming Bartholomew the villain.
CONTINUITY NOTES: Bartholomew is her brother.

--- SCRIPT START ---

SCENE 01
SCENE HEADING: INT. HALL - NIGHT
SCENE START: 00:00.00
SCENE END: 00:40.00

ACTIONS AND DIALOGUE:

**CAMERA:**
Start: (00:00.00) | End: (00:06.00)
Slow establishing pan across the empty hall.

**ACTION:**
Start: (00:06.00) | End: (00:08.00)
Bartholomew slaps Mira across the face.

**MIRA (Shocked, Furious):**
Start: (00:08.10) | End: (00:10.00)
"You stole my wedding!"

**BARTHOLOMEW (O.S.) (Cold, Cruel):**
Start: (00:10.20) | End: (00:13.00)
"Your husband is dead, sister."

**CAMERA:**
Start: (00:13.00) | End: (00:30.00)
Extreme close-up on Mira's face as she trembles.

**MIRA (Desperate):**
Start: (00:14.00) | End: (00:29.00)
"Who killed him? Tell me! Who?"

**SOUND FX:**
Start: (00:29.00) | End: (00:30.00)
Ominous drone swells to a cliffhanger peak.

**CAMERA:**
Start: (00:30.00) | End: (00:31.00)
Cut to black.

--- SCRIPT END ---
"""


def test_timestamp_math():
    assert ts_to_sec("01:18.50") == 78.5
    assert ts_to_sec("00:00.00") == 0.0
    assert ts_to_sec("01:02:03.50") == 3723.5


def test_parser_uppercase_scene_fields_and_qualifiers():
    ep = parse_text(MINI, "episode-007-x.txt")
    assert ep.episode_number == 7 and len(ep.scenes) == 1 and ep.scenes[0].end_sec == 40.0
    d = [e for e in ep.events if e.track == "DIALOGUE"]
    assert d[1].speaker_raw == "BARTHOLOMEW" and d[1].qualifiers == ["O.S."] and d[1].emotion_tags == ["Cold", "Cruel"]
    assert d[0].text == "You stole my wedding!"


def test_wall_clock_not_block_sum():
    ep = parse_text(MINI, "episode-007-x.txt")
    w = slice_window(EpisodeIndex(ep), 6.0, 30.0)
    assert w.duration == 24.0
    assert w.contained_block_sum > w.duration          # the naive number is wrong


def test_cut_to_black_excluded_from_ends():
    ix = EpisodeIndex(parse_text(MINI, "episode-007-x.txt"))
    assert all(t <= 30.0 for t, _ in ix.clean_ends)


def test_blind_prompt_has_no_header_or_unspoken_names():
    ep = parse_text(MINI, "episode-007-x.txt")
    gen = CandidateGenerator([ep], "Test Show")
    cands = gen.generate_all()
    assert cands, "expected at least one candidate"
    ev = PromoEvaluator(gen, LLMAdapter("heuristic"), "Test Show")
    prompt = ev.render_prompt(cands[0])
    assert "Secret logline" not in prompt and "brother" not in prompt
    assert "Bartholomew" not in prompt and "BARTHOLOMEW" not in prompt   # never spoken -> masked
    assert "SPEAKER A" in prompt


def test_explicit_provider_without_key_raises():
    from src.llm_adapter import MissingAPIKeyError
    saved = os.environ.pop("GEMINI_API_KEY", None)
    try:
        import src.config as cfg
        orig = cfg.load_dotenv_if_present
        cfg.load_dotenv_if_present = lambda: None        # ignore a real .env for this test
        try:
            LLMAdapter("gemini")
            raise AssertionError("expected MissingAPIKeyError")
        except MissingAPIKeyError as exc:
            assert "GEMINI_API_KEY" in str(exc)
        finally:
            cfg.load_dotenv_if_present = orig
    finally:
        if saved is not None:
            os.environ["GEMINI_API_KEY"] = saved


def test_mock_gemini_server_rerank_pacing_429_backoff_and_cache():
    """Real HTTP path of stage 2 against a local mock of generateContent: the 2nd request gets a 429; a re-run
    is answered from the cache with no request at all."""
    import json
    import shutil
    import tempfile
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from src.ranker import shortlist
    from src.reranker import ModelReranker
    from src.show_brief import offline_brief
    seen = {"n": 0, "bodies": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["n"] += 1
            seen["bodies"].append(body)
            if seen["n"] == 2:
                msg = json.dumps({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"}]}}).encode()
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(msg)
                return
            user = body["contents"][0]["parts"][0]["text"]
            ids = re.findall(r"### CLIP (C\d\d)", user)
            clips = [{"id": i, "score": 90 - 5 * k, "why": "SPEAKER A is cornered and the clip cuts before the answer.",
                      "sells_premise": False} for k, i in enumerate(ids)]
            out = {"candidates": [{"content": {"parts": [{"text": json.dumps({"clips": clips})}]}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 1000 + len(user) // 4, "candidatesTokenCount": 40 * len(ids),
                                     "thoughtsTokenCount": 300}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    os.environ["GEMINI_BASE_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    cache, orig = Path(tempfile.mkdtemp()), config.LLM_CACHE_DIR
    config.LLM_CACHE_DIR = cache
    try:
        words = ("wedding", "ring", "house", "name", "child", "crown")       # six different clips, six cache keys
        eps = [parse_text(MINI.replace("Episode 7", f"Episode {n}").replace("my wedding", f"my {w}"),
                          f"episode-00{n}-x.txt") for n, w in zip((1, 2, 3, 4, 5, 6), words)]
        brief = offline_brief(eps, "Test Show")
        gen = CandidateGenerator(eps, "Test Show", brief=brief)
        cands = gen.generate_all()
        ev = PromoEvaluator(gen, "Test Show", brief=brief)
        promos = ev.evaluate(cands)
        shortlist(promos, size=1000)
        logs = []
        ad = LLMAdapter("gemini", "gemini-2.5-pro", api_key="test-key", pacing_sec=0.2,
                        backoff_schedule=(0.3, 0.6, 1.2, 2.4), log=logs.append)
        rr = ModelReranker(ad, ev, brief, group_size=2)
        t0 = time.time()
        out = rr.rerank(promos, progress=logs.append)
        n_groups = len(rr.groups(promos))
        assert all(r.by == "gemini:gemini-2.5-pro" for r in out.values())          # no fallback
        assert ad.meter.calls == n_groups and ad.meter.rate_limit_retries == 1
        assert seen["n"] == n_groups + 1                                             # one extra = the 429 retry
        assert ad.meter.backoff_wait_s >= 0.3 and ad.meter.pacing_wait_s > 0
        assert ad.meter.thinking_tokens == 300 * n_groups and ad.meter.cost_usd > 0
        assert any("rate-limited" in m for m in logs)
        cfg = seen["bodies"][0]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json" and "thinkingConfig" in cfg
        for b in seen["bodies"]:
            txt = json.dumps(b)
            assert "Secret logline" not in txt and "EPISODE LOGLINE" not in txt and "PROMO_EP" not in txt
        assert time.time() - t0 >= 0.2 * (n_groups - 1)
        before = seen["n"]
        ad2 = LLMAdapter("gemini", "gemini-2.5-pro", api_key="test-key", pacing_sec=0.2)
        rr2 = ModelReranker(ad2, ev, brief, group_size=2)
        t1 = time.time()
        rr2.rerank(promos)
        assert seen["n"] == before and rr2.stats["cached_groups"] == n_groups and time.time() - t1 < 1.0
    finally:
        srv.shutdown()
        os.environ.pop("GEMINI_BASE_URL", None)
        config.LLM_CACHE_DIR = orig
        shutil.rmtree(cache)


def test_full_corpus_invariants():
    if config.DEFAULT_INPUT_DIR is None:
        return
    eps, errs = load_corpus(config.DEFAULT_INPUT_DIR)
    assert not errs and len(eps) >= 1
    title = config.show_title_from_dir(config.DEFAULT_INPUT_DIR)
    gen = CandidateGenerator(eps[:6], title)
    promos = rank(PromoEvaluator(gen, LLMAdapter("heuristic"), title).evaluate(gen.generate_all()))
    raw = {e.episode_number: Path(e.path).read_text(encoding="utf-8") for e in eps}
    for p in promos:
        assert 20.0 <= p.total_duration_sec <= 60.0
        for s in p.segments:
            assert f"({s['start_timestamp']})" in raw[s["episode"]] and f"({s['end_timestamp']})" in raw[s["episode"]]
    assert {p.promo_type for p in promos} >= {"continuous", "multi_segment_intra"}


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # pragma: no cover
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
