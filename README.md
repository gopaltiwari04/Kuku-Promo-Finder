# Find the Promos

Kuku AI Generalist Assignment, **Problem 1** · Gopal Krishna Tewari · B.Tech CSAM · Roll No. 2023223

The tool reads every episode screenplay of a microdrama and finds the 20–60 s clips worth posting as ads on
Instagram, TikTok or YouTube. For each promo it gives the episode and start–end timestamps of every segment, the
hook, the cliffhanger and one line on why it works. It sorts them into what to post first, next and later, and it
reports where it picks badly.

The two-page write-up and the run on the full input pack (`raw_outputs/`) are shared by link with the submission,
not in this repo: they quote Kuku's confidential screenplay (see [`DEPLOY.md`](DEPLOY.md)). Running the steps below
on the input pack recreates both; `scripts/build_doc.py` writes the doc from `raw_outputs/`, and a test fails if the
two disagree.

## Quick start

```
pip install -r requirements.txt
python run_pipeline.py              # all episodes -> raw_outputs/ (about 12 s, free, no API key)
streamlit run app.py                # review screen; reads raw_outputs/
python -m pytest -q                 # tests
python scripts/build_doc.py         # rebuild the 2-page doc from raw_outputs/
```

The episode folder (any folder of `episode-NNN-*.txt` screenplays; its name is the show title) is found automatically and is **read-only**: every
run checks that each source file's sha256 is unchanged. It is git-ignored and never committed.

## The method: code shortlists, a re-ranker orders

```
 episode .txt files (read-only)
   STAGE 1 - CODE (free, same answer every run)
     read every block; learn the cast and a show brief (premise, twist,     parser.py, show_profile.py,
       this show's own stakes words) from the title and the script          show_brief.py
     cut candidate clips that start 0.3 s before a line or action and      candidate_generator.py, scoring.py
       end on an open beat (stretches, jump-cuts, arcs, premise montages)
     rate each clip blind, write the copy from its own lines, hold the     evaluator.py, ranker.shortlist()
       clips that fail a test, shortlist the best ~100
   STAGE 2 - RE-RANKER (swappable)
     order the shortlist and write the why-it-works line                   reranker.py (+ llm_adapter.py)
   CODE   build Post first / Post next / Later / Hold with the rules        ranker.py
   CODE   audit bad picks, run the checks, measure quality                  quality_auditor.py, reporting.py,
                                                                            quality_eval.py
   then a PERSON spends about 10 minutes in the app: watch, approve or reject, nudge a cut, export
```

**Code always picks the timestamps.** Every cut sits on a verified screenplay block (a tight start sits 0.3 s before
one), so a model cannot break a duration or a timestamp, or let a clip that failed a test back in.

**Nothing in the code is about this show.** The word lists are genre-general (family, betrayal, money, threats). The
show's own vocabulary is learned per run: the title gives the premise and the twist, and the stakes words are the
words that are over-represented in the script's threat, accusation and reveal lines. With a model, one cached call writes the same brief, checked against the script.

**Two re-rankers plug in.** The default is `keyword` (free, offline): the order is the stage-1 score and the line is
written by code. Any model can replace it with one flag: it scores the shortlist 0-100 in groups of 10 and writes the
line; code keeps a line only if every quote in it is spoken in the clip. Every answer is cached in `.llm_cache/`, a
failing or slow group keeps the keyword order, and a time budget caps the stage, so a live demo never stalls.

```
cp .env.example .env                # add GEMINI_API_KEY=... (never commit .env)
python run_pipeline.py --provider gemini --model gemini-2.5-flash --output_dir runs/gemini
python run_pipeline.py --provider openai --model <id>   # OPENAI_BASE_URL=... for any OpenAI-compatible endpoint
python run_pipeline.py --provider gemini --cache_only   # replay saved answers: no key, no network
```

## The posting rules (all in `src/config.py`, each with its reason)

| Setting | Value | Why |
|---|---|---|
| `min_hook` | 6 / 10 | brief test 1: the first 3 s must stop a scroll; checked on the scorer's rating and on code's own |
| open ending | required | brief test 2: a clip whose last line, shots or sound settle the scene is held |
| `min_clarity` | 6 / 10 | brief test 3: below this a stranger is lost |
| `min_score` | 50 / 100 | half marks; below that it is not worth a post |
| `post_first` | 5 | a marketer can launch five ads on day one |
| `first_quarter` | 25% of episodes | Post first comes from early episodes, so a click lands near the start |
| `first_max_spoiler` / `next_max_spoiler` | 3.5 / 5 of 10 | Post first gives nothing away; Post next a little more |
| `post_next` / `later` | 10 / up to 25 | the next week of tests, then a retargeting pool |
| `max_per_sequence` | 2 per list | not one scene cut ten ways |
| `chain_gap_sec` | 3 s | a clip starting within 3 s of another's cut continues it, so it becomes an alternate |
| `premise_slots` | 1 | one Post first slot is kept for the best clip that sells the title's twist |
| `FINALE_SHARE` | last 10% of episodes | the finale is never a cold promo: Later only, marked retarget |
| `PRE_ROLL_SEC` | 0.3 s | every clip starts this long before its first line or action |
| `SHORTLIST_SIZE` | 100 | how many stage-1 clips the re-ranker sees |

Stage-1 score (0–100) = 30% hook + 25% cliffhanger + 25% makes sense alone + 10% pacing + 10% virality − 12%
spoiler. Inside each list, order = the stage-2 score (the stage-1 score with the keyword re-ranker). No two posts
share footage.

## What you get (`raw_outputs/`)

| File | What is in it |
|---|---|
| `ranked_promos.csv` | the posting plan in order: list, segments, hook, cliffhanger, why it works, scores, reason |
| `held_promos.csv` | promos found but not posted, each with the reason (failed a test, or an alternate of a post) |
| `promos_all_episodes.json` | every promo with all fields and scores |
| `ranked_promos_report.md` | the plan as a readable report |
| `quality_audit_and_bad_picks.json` | failure modes found on this show, with before/after and short evidence |
| `quality_eval.json` / `.md` | repetition, model re-rank vs code score, comparison with real ads, rule checks |
| `episode_summary_stats.csv` | per-episode numbers |
| `run_metadata.json`, `verification_report.json` | settings, learned cast, cost estimate, and the automatic checks |

Saved outputs keep only the hook and cut lines from the screenplays (no transcripts, no full prompts).

## How quality is measured

1. **Blind A/B test** (`scripts/blind_eval.py`): the 15 promos of Post first and Post next against 15 baselines, the
   last 40 s of a random non-finale episode. 30 clips are shuffled with the source hidden; judges score hook, ending
   and makes sense alone (1-5) and would-post (Y/M/N).
   ```
   python scripts/blind_eval.py packet                       # eval/packet.md, key.json, manual_labels.csv, JUDGE_PROMPT.md
   python scripts/blind_eval.py judge --provider gemini      # Judge 1 -> eval/judge1_<model>.json (cached)
   #   Judge 2: paste eval/JUDGE_PROMPT.md + eval/packet.md into another model, save its JSON as eval/judge2_<model>.json
   #   Person:  fill eval/manual_labels.csv
   python scripts/blind_eval.py merge                        # eval/results.md: means, difference, win rate, agreement
   python scripts/build_doc.py                               # puts the results in the 2-page doc
   ```
   `eval/packet.*` hold screenplay text and are git-ignored; the key, the judges' answers and the results are not.
2. **Repetition**: posts per story sequence in each list.
3. **Model vs code**: with a model re-ranker, its order next to the stage-1 score on the shortlist.
4. **Real ads**: the plan next to published teardowns of ReelShort and DramaBox ads (sources below).
5. **Earlier blind labels** (`labels/`): an AI judge's labels from while the rules were being fixed, matched to
   every run by `quality_eval`.
6. **Rule checks**: the automatic checks in `verification_report.json` (timestamps verbatim in the source, 20–60 s,
   no repeats, a premise clip in Post first, no finale in the cold lists, files unchanged). They show the rules ran,
   not that the picks are good.

## The review screen (`app.py`)

1. **Promos:** every promo with segments, hook, cliffhanger, why it works and scores; approve, reject, move a cut by
   one block, add a note, download the approved cut list.
2. **What to post first:** the rules in plain words and the three lists with a reason for each promo.
3. **Where it picks badly:** the bad picks it found, repetition, the real-ad comparison and the failure modes with
   evidence.
4. **Try a new screenplay:** upload or paste episodes and get results in the same format.

Decisions are saved to `review/decisions.json` (git-ignored). The header shows the Kuku FM logo if an image named
`kuku_logo.*` is anywhere under `static/`.

## A new show

```
python run_pipeline.py --input_dir "path/to/new show" --output_dir runs/new_show
```

Nothing about this show is hard-coded: the cast, lead, premise and stakes words are learned from the screenplays and
the title. A test
runs the tool on a 3-episode CEO drama written in a different screenplay format (`tests/fixtures/`). Optionally add
`profiles/<show-name>.json` to merge descriptor speakers ("THE MASKED STRANGER") with named characters; profile files
name the cast, so they are git-ignored.

## Cost

The keyword re-ranker is free. A model re-rank is about 8 calls per show (one for the show brief, the rest for the
shortlist in groups of 10); the estimate for each model is printed at the end of a run and stored in
`run_metadata.json` (gemini-3.5-flash-lite is about $0.03 per show at list price). A re-run is answered from the
cache: no cost, no wait.

## Sources for the real-ad comparison

* [AdMapix, ReelShort ads 2026](https://www.admapix.com/blog/ad-intelligence/reelshort-ad-strategy-2026): 8 US ads
  at 0/2/5/10 s (openings, lengths, reuse of one video across many ad records).
* [AdMapix, DramaBox ad strategy 2026](https://www.admapix.com/blog/ad-intelligence/dramabox-ad-strategy-2026): the
  US top 10 (lengths, story types, share of impressions).
* [invideo, micro-drama hooks and cliffhangers](https://invideo.io/blog/ai-micro-drama-hooks-cliffhangers/): a 3 s
  hook, and cut one beat before the resolution.
* [TheWrap, DramaShorts co-founder interview](https://www.thewrap.com/dramashorts-co-founder-leo-ovdiienko-guilty-pleasure-viewing-interview):
  ads take the hook and the most provocative parts, then lead viewers into the app.
* [SocialPeta weekly ranking, Feb 2026](https://socialpeta.com/en/blog/short-drama-weekly-rankin-how-phased-ad-strategies-unlock-viral-growth):
  hidden identity, public ridicule and revenge in top ads.

These are third-party estimates; they describe what top ads look like, not what makes them convert.

## Limits

* Text cannot show acting, voice or effects, so a person watches the plan on video before posting.
* The hook and clarity scores read words and names, not stakes; clarity is strict and holds some clips a person
  would post (`held_promos.csv` lists each with its reason).
* In the A/B packet a jump cut is visible, so a judge can tell a montage from a baseline.
* The brief caps promos at 60 s; DramaBox's biggest ads run about 2 minutes, a format this tool does not make.
* The rules are set by judgment; real post performance is the next step to tune them.

See [`DEPLOY.md`](DEPLOY.md) to share the app.
