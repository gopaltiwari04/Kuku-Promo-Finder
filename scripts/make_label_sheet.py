"""Make a blind labelling sheet: 40 promos (the first 10 of the posting plan, 10 more from the plan, 20 the rules held
for failing a test), shuffled, shown only as their screenplay lines - no scores, no list, no tool copy.

    python scripts/make_label_sheet.py            # writes labels/label_sheet.csv + labels/label_key.csv

Two people fill in rater1_would_post / rater2_would_post with y or n ("would I post this as an ad?").
Then: python scripts/quality_eval.py   -> precision@10, plan precision, agreement (kappa).
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.parser import EpisodeIndex, load_corpus, ts_to_sec  # noqa: E402

out_dir = config.DEFAULT_OUTPUT_DIR
promos = json.loads((out_dir / "promos_all_episodes.json").read_text(encoding="utf-8"))["promos"]
plan = [p for p in sorted(promos, key=lambda p: p["post_order"]) if p.get("wave") in (1, 2, 3)]
held = [p for p in promos if p.get("wave") == 4 and p.get("wave_reason", "").startswith("held:")
        and "plan is full" not in p.get("wave_reason", "")]          # failed a test (not an alternate of a post)
rng = random.Random(7)
sample = plan[:10] + rng.sample(plan[10:], min(10, len(plan[10:]))) + rng.sample(held, min(20, len(held)))
rng.shuffle(sample)
eps, _ = load_corpus(config.DEFAULT_INPUT_DIR)
idx = {e.episode_number: EpisodeIndex(e) for e in eps}


def clip_text(p):
    lines = []
    for n, s in enumerate(p["segments"], 1):
        a, b = ts_to_sec(s["start_timestamp"]), ts_to_sec(s["end_timestamp"])
        lines.append(f"--- segment {n}: Ep {s['episode']} {s['start_timestamp']}-{s['end_timestamp']} ---")
        for e in idx[s["episode"]].events_in(a, b):
            if e.track == "DIALOGUE":
                lines.append(f"{e.speaker_raw} ({', '.join(e.emotion_tags)}): \"{e.text}\"")
            else:
                lines.append(f"[{e.track}] {e.text}")
    return "\n".join(lines)


labels = ROOT / "labels"
labels.mkdir(exist_ok=True)
with open(labels / "label_sheet.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["item", "length_sec", "clip_text", "rater1_would_post", "rater1_note", "rater2_would_post", "rater2_note"])
    for i, p in enumerate(sample, 1):
        w.writerow([f"{i:02d}", f"{p['total_duration_sec']:.1f}", clip_text(p), "", "", "", ""])
with open(labels / "label_key.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["item", "promo_id"])
    for i, p in enumerate(sample, 1):
        w.writerow([f"{i:02d}", p["promo_id"]])
print(f"wrote {len(sample)} items to labels/label_sheet.csv (key in labels/label_key.csv; do not open it while labelling)")
