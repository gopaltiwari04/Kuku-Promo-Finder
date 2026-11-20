"""Re-run the quality evaluation of an output folder, optionally comparing with other runs.

    python scripts/quality_eval.py                                   # raw_outputs/ (also runs after every pipeline run)
    python scripts/quality_eval.py --compare runs/gemini_rerun       # stability: same model, run again with --no_cache
    python scripts/quality_eval.py --compare runs/gemini_2_5_flash   # second model on the same candidates
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.quality_eval import evaluate_run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=config.DEFAULT_OUTPUT_DIR)
ap.add_argument("--compare", type=Path, nargs="*", default=[])
ap.add_argument("--labels", type=Path, default=ROOT / "labels")
a = ap.parse_args()
rep = evaluate_run(a.out, a.compare, a.labels)
print((a.out / "quality_eval.md").read_text(encoding="utf-8"))
