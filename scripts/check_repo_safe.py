"""Before pushing anywhere: list what would be committed and stop if any confidential or secret file is in it.

    python scripts/check_repo_safe.py

Fails (exit 1) on: episode screenplays (files containing the screenplay markers), .env, Streamlit secrets,
API-key-looking strings, any output folder that quotes the show (raw_outputs/, docs/, eval answers, profiles/),
and - when the input pack is on this machine - any file that names the show's title or its cast.
"""
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_RX = re.compile(r"(AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_\-]{20,})")


def candidate_files():
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.splitlines()
        if out:
            return [ROOT / f for f in out]
    except Exception:
        pass
    pats = [l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    keep = [p[1:].strip("/") for p in pats if p.startswith("!")]          # negations, e.g. !labels/judge_*.csv
    pats = [p for p in pats if not p.startswith("!")]
    dir_pats = [p.strip("/") for p in pats if p.endswith("/")]
    file_pats = [p.strip("/") for p in pats if not p.endswith("/")]
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune ignored folders (.venv, the episode folder, caches) instead of walking them
        dirnames[:] = [d for d in dirnames if d != ".git" and not any(fnmatch.fnmatch(d, pt) for pt in dir_pats)]
        for name in filenames:
            f = Path(dirpath) / name
            rel = f.relative_to(ROOT).as_posix()
            ignored = any(fnmatch.fnmatch(rel, pt) or fnmatch.fnmatch(name, pt) for pt in file_pats)
            if not ignored or any(fnmatch.fnmatch(rel, pt) for pt in keep):
                files.append(f)
    return files


def show_words():
    """The input pack's title and cast names (learned, never typed in), to make sure no public file names them."""
    sys.path.insert(0, str(ROOT))
    try:
        from src import config
        from src.parser import load_corpus
        from src.show_profile import profile_for
    except Exception:
        return "", set()
    if config.DEFAULT_INPUT_DIR is None:
        return "", set()
    eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
    title = config.show_title_from_dir(config.DEFAULT_INPUT_DIR)
    prof = profile_for(eps, title)
    names = {w for x in prof.core_cast | prof.protagonists | prof.premise_cast for w in x.split() if len(w) >= 4}
    return title, names - {"LADY", "LORD", "KING", "QUEEN", "DUKE", "MISTER", "MADAM"}


QUOTING = ("raw_outputs/", "docs/", "profiles/", "eval/judge", "eval/manual", "eval/results", "eval/packet")
bad = []
files = candidate_files()
title, names = show_words()
name_rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b", re.I) if names else None
for f in files:
    rel = f.relative_to(ROOT).as_posix()
    if f.name == ".env" or rel.endswith("secrets.toml"):
        bad.append(f"{rel}: secrets file")
        continue
    try:
        txt = f.read_text(encoding="utf-8", errors="ignore") if f.stat().st_size < 5_000_000 else ""
    except Exception:
        txt = ""
    synthetic = rel.startswith("tests/fixtures/") and txt.startswith("SYNTHETIC TEST SHOW")
    if "--- SCRIPT START ---" in txt and f.suffix == ".txt" and not synthetic:
        bad.append(f"{rel}: looks like an episode screenplay (confidential input pack)")
    if KEY_RX.search(txt):
        bad.append(f"{rel}: contains an API-key-looking string")
    if rel.startswith(QUOTING):
        bad.append(f"{rel}: output that quotes or describes the show (share it by link, not in the repo)")
        continue
    if rel.startswith("scripts/check_repo_safe"):
        continue
    if title and title.lower() in txt.lower():
        bad.append(f"{rel}: names the show's title")
    if name_rx is not None:
        m = name_rx.search(txt)
        if m:
            bad.append(f"{rel}: names a character from the show ({m.group(1)})")
print(f"{len(files)} files would be committed." + (f" Checked for the show's title and {len(names)} cast names."
                                                    if names else ""))
if bad:
    print("STOP - remove these before pushing:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("OK - no screenplays, show names, quoted outputs, secrets or keys.")
