"""Project configuration, auto-discovery of the episode folder, and tunable defaults.

Everything a reviewer might want to change in an interview lives here: duration bounds,
ranking weights, wave rules, provider/model defaults and token prices. (The cast of each show is learned by
src/show_profile.py, not configured here.)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EPISODE_FILE_RE = re.compile(r"^episode-(\d{1,4})", re.I)
EXCLUDED_DIRS = {"raw_outputs", "docs", "src", "tests", "scripts", "static", "eval", "labels", "profiles", "runs",
                 "review", ".git", ".venv", "venv", "__pycache__", "node_modules", ".streamlit"}


def _looks_like_episode(path: Path) -> bool:
    if not path.name.lower().endswith(".txt"):
        return False
    if EPISODE_FILE_RE.match(path.name):
        return True
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            head = fh.read(400)
        return "EPISODE NUMBER" in head.upper()
    except OSError:
        return False


def count_episode_files(folder: Path) -> int:
    try:
        return sum(1 for p in folder.iterdir() if p.is_file() and _looks_like_episode(p))
    except OSError:
        return 0


def discover_episode_dir(root: Path = PROJECT_ROOT, max_depth: int = 3) -> Optional[Path]:
    """Return the folder (up to 3 levels down) holding the most episode screenplays. Its name is the show title."""
    best: Optional[Path] = None
    best_n = 0
    for dirpath, dirnames, _ in os.walk(root):
        rel_depth = len(Path(dirpath).relative_to(root).parts)
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        if rel_depth >= max_depth:
            dirnames[:] = []
        n = count_episode_files(Path(dirpath))
        if n > best_n:
            best, best_n = Path(dirpath), n
    return best


DEFAULT_INPUT_DIR: Optional[Path] = discover_episode_dir()
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "raw_outputs"


def show_title_from_dir(input_dir: Optional[Path]) -> str:
    """The folder name is the show title (e.g. 'My Secret Husband Is the CEO')."""
    return input_dir.name if input_dir else "Untitled Show"


# --------------------------------------------------------------------------------------
# Promo geometry
# --------------------------------------------------------------------------------------
MIN_DURATION = 20.0
MAX_DURATION = 60.0
HOOK_WINDOW_SEC = 3.0            # the scroll-stopper window
CLIFF_WINDOW_SEC = 4.0           # the final beat inspected for the cut-out
SWEET_SPOT = (24.0, 50.0)        # mild preference band inside [20, 60]

# Type B: intra-episode jump cut  (Segment 1 hook/setup + Segment 2 climax/cliffhanger)
INTRA_SEG1 = (12.0, 22.0)
INTRA_SEG2 = (18.0, 36.0)
INTRA_TOTAL = (24.0, 58.0)
INTRA_MIN_SKIP = 6.0             # must skip at least this much middle material

# Type C: cross-episode arc supercut  (early wound + later escalation/payback tease)
CROSS_SETUP = (12.0, 25.0)
CROSS_PAYOFF = (15.0, 35.0)
CROSS_TOTAL = (25.0, 58.0)
CROSS_MIN_EP_GAP = 2

# Candidate budget (per episode)
MAX_CONTINUOUS_PER_EP = 3
MAX_INTRA_PER_EP = 1
MAX_CROSS_TOTAL = 18
NMS_IOU = 0.30
# Two stages: code shortlists, then a re-ranker (keyword or model) orders the shortlist and writes the copy
SHORTLIST_SIZE = 100                          # stage 1 hands this many candidates to stage 2
PRE_ROLL_SEC = 0.3                            # a clip starts this long before its first line or action
FINALE_SHARE = 0.10                           # the last 10% of episodes never appear in cold promos
PREMISE_CLARITY_BONUS = 1.0                   # a clip that states the title's twist: the title supplies the context
RERANK_GROUP_SIZE = 10                        # clips per model call (each group spans the whole score range)
RERANK_TIME_BUDGET_SEC = 240.0                # after this, remaining groups use the keyword re-ranker
LLM_TIMEOUT_SEC = 60.0                        # per HTTP call
LLM_PACING_SEC = 4.0                          # gap between consecutive model calls (free-tier RPM)
RATE_LIMIT_BACKOFF_SEC = (10.0, 20.0, 40.0)   # 429 / RESOURCE_EXHAUSTED retries
LLM_MAX_OUTPUT_TOKENS = 8192
GEMINI_THINKING_BUDGET = 1024                 # caps 2.5-series "thinking" tokens per call (billed as output)
LLM_CACHE_DIR = PROJECT_ROOT / ".llm_cache"   # content-addressed answers: re-runs and live demos never wait

# --------------------------------------------------------------------------------------
# Ranking formula  (Stage 4, pure code)
#   Raw = 0.30 Hook + 0.25 Cliff + 0.25 Clarity + 0.10 Pacing + 0.10 Virality - 0.12 Spoiler
#   Composite = round(clip(Raw * 10, 0, 100), 2)
# --------------------------------------------------------------------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "hook_strength": 0.30,
    "cliffhanger_tension": 0.25,
    "standalone_clarity": 0.25,
    "pacing_density": 0.10,
    "trope_virality": 0.10,
    "spoiler_risk": -0.12,
}

WAVE_LABELS = {
    1: "Post first",
    2: "Post next",
    3: "Later",
    4: "Hold",
}
# The whole posting plan in nine settings. Each says where its number comes from.
POSTING_RULES = {
    "post_first": 5,              # a marketer can launch five ads on day one ("which promos would you post first")
    "post_next": 10,              # the next week of tests
    "later": 25,                  # retargeting pool; the plan never grows past 40 posts
    "first_quarter": 0.25,        # Post first comes from the first quarter of the show: a click lands near the start
    "first_max_spoiler": 3.5,     # 0-10; above this a stranger learns a later twist (scorer's spoiler scale)
    "next_max_spoiler": 5.0,      # Post next allows a little more; anything higher goes to Later with a check
    "min_hook": 6.0,              # brief test 1: the first 3 s must stop a scroll; a line that merely exists is not
                                  # enough. Checked on the scorer's rating AND on the code's own (a second opinion)
    "min_clarity": 6.0,           # brief test 3, "makes sense on its own": below 6/10 a stranger is lost
    "min_score": 50.0,            # half marks on the 100-point score; below that it is not worth a post
    "max_per_sequence": 2,        # at most 2 posts from one story sequence per list: not one scene cut ten ways
    "chain_gap_sec": 3.0,         # a clip starting within 3 s of another's cut continues it -> alternate
    "premise_slots": 1,           # one title reveal ("You're the CEO?") may join Post first from any episode
    "human_spoiler_check_above": 5.0,
}

# --------------------------------------------------------------------------------------
# Providers (Stage 3) — swap with --provider / --model, keys come from .env
# --------------------------------------------------------------------------------------
PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "heuristic": "heuristic-v1",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    # Any model id can be passed with --model.
    "anthropic": "claude-sonnet-5-5",
}
PROVIDER_ENV_KEYS: Dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
# USD per 1M tokens (input, output). List prices at time of writing — edit if they change.
# USD per 1M (input, output) tokens at list price, checked 2026-11-19. Free tiers (e.g. Google AI Studio) cost $0
# within their rate limits; the pipeline always reports the list-price equivalent so costs at scale are honest.
MODEL_PRICES_PER_M: Dict[str, Tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),    # ai.google.dev pricing; output includes thinking tokens
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),          # prompts <= 200k tokens; output price includes thinking
    "gpt-4o-mini": (0.15, 0.60),              # OpenAI path not run live in this project (mock-tested only)
    "claude-sonnet-5-5": (2.00, 10.00),       # platform.claude.com/docs/en/about-claude/pricing
    "claude-haiku-5-5": (0.10, 0.50),         # prompts <= 100k tokens (each batch here is ~10-15k)
    "heuristic-v1": (0.0, 0.0),
}
CHARS_PER_TOKEN = 4.0

# Cast (lead, recurring characters, title character, aliases) is learned per show by src/show_profile.py;
# an optional human-reviewed file can be dropped in profiles/<show-slug>.json.


def load_dotenv_if_present() -> None:
    """Load .env without making python-dotenv a hard dependency."""
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path)
        return
    except Exception:
        pass
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
