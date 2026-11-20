# Labels

The main quality evidence is the blind A/B test in `eval/` (see the README). The files here are earlier
labels, still matched to every run by `quality_eval`.

## Second judge (committed): `judge_labels.csv`

Blind labels from an AI judge for 105 clips: every post in the shipped plan, the first lists of the two runs
before the fixes, and a sample of promos the rules held back. The judge read each clip as screenplay lines only (no
episode number, score, rank, list or tool copy), in shuffled packets, and gave four labels:

| Column | Values |
|---|---|
| `hook` | S strong, W weak, F fails (would the first 3 s stop a scroll?) |
| `ending` | O open, W soft, R resolved |
| `makes_sense_alone` | Y, P partly, N |
| `would_post` | Y, M maybe, N |
| `round` | 1-3 labelled while the rules were being fixed; 4 after the main fixes; 5 the posts the last fix brought in |
| `before_plans` | where the clip sat in an earlier plan, e.g. `before_gemini:wave1#3` |

Clips are identified by timings only (`Ep12 00:26.80-01:09.20`), so the file holds no screenplay text.
`python scripts/quality_eval.py` matches it to the current plan. Caveat: the judge is an AI model, not a person, and not
independent of the tools used to build this. It is a second opinion, not ground truth, which is why the human sheet
below exists.

## Human labels (git-ignored): `label_sheet.csv`

1. `python scripts/make_label_sheet.py` writes `label_sheet.csv` (40 promos as screenplay lines: the first 10 of the
   plan, 10 more from the plan, 20 the rules held for failing a test, shuffled) and `label_key.csv`.
2. Two people each read every item and write `y` or `n` in their column: *would I post this as an ad for the show?*
   Do not open `label_key.csv` or the outputs while labelling.
3. `python scripts/quality_eval.py` reports precision@10, plan precision, how many held promos people also reject,
   and agreement (Cohen's kappa) between the tool and people and between the two people.

The sheet contains screenplay text from the input pack, so it is not committed.
