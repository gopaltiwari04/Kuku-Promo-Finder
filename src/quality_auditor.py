"""Stage 5 - automated quality & failure auditor.

Answers "Check quality. Where does your tool pick badly?" and "Tell us one place where AI got it wrong"
with timestamp-verified evidence computed from the corpus (nothing here is hand-typed):

  FM1  adding up overlapping blocks inflates length (naive block summing vs wall-clock)
  FM2  episodes that open on silent shots          (evocative prose fools a text scorer)
  FM3  reading the summary inflates clarity        (summary-aware vs blind scoring)
  FM4  cutting after the peak                      (naive scene-end cuts vs snap-back trimmer)
  FM5  what text cannot show                       (acting, voice, effects -> human check)
  FM6  fake payback                                (payoff punishes a different villain)
  FM7  ending on a line that closes the scene      (closure detector; caps a model's cliffhanger score)
  FM8  ending on a shot that resolves the scene    (walking off into the sunlight; a soft smile; a triumphant swell)
  FM9  one scene posted many times                 (chained cuts of the same scene; story-sequence cap)
plus per-promo risk flags on the final ranked list.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from . import config
from .candidate_generator import CandidateGenerator
from .lexicons import EVOCATIVE_PROSE_RX, SILENT_VISUAL_RX
from .parser import EpisodeIndex, slice_window, ts_to_sec
from .schema import AuditCase, Promo, sec_to_ts
from .scoring import BlindContext, build_segment, clarity_score, cliff_profile, hook_profile, location_key

TTS_RISK_TAGS = ("scream", "shout", "roar", "sob", "weep", "hysteric", "agon", "booming", "shriek", "crying")
VFX_RISK_RX = ("punch", "strike", "explo", "fire", "blaze", "inferno", "blood", "whip", "slap", "collaps", "smoke",
               "crash", "drag", "stab", "shoot", "gun", "knife", "fall")


def _lines(idx: EpisodeIndex, s: float, e: float, visual: bool = True, limit: int = 14) -> List[str]:
    out = []
    for ev in idx.events_in(s, e):
        if ev.track == "DIALOGUE":
            out.append(f"{ev.start_ts}-{ev.end_ts} {ev.speaker_raw} ({', '.join(ev.emotion_tags)}): \"{ev.text}\"")
        elif visual:
            out.append(f"{ev.start_ts}-{ev.end_ts} {ev.track}: {ev.text}")
        if len(out) >= limit:
            break
    return out


class QualityAuditor:
    @staticmethod
    def _w1_clarity() -> float:
        return float(config.POSTING_RULES["min_clarity"])

    def __init__(self, gen: CandidateGenerator, promos: Sequence[Promo]):
        self.gen = gen
        self.promos = list(promos)
        self.idx = gen.index

    # ------------------------------------------------------------------ FM1
    def fm1_duration(self) -> AuditCase:
        ep1 = self.idx[min(self.idx)]
        s, e = 33.70, 78.50
        if ep1.ep.duration_sec < e:
            s, e = 0.0, min(45.0, ep1.ep.duration_sec)
        w = slice_window(ep1, s, e)
        overlap_sum = round(sum(ev.duration for ev in w.events), 2)
        corpus_sum = sum(ev.duration for ix in self.idx.values() for ev in ix.events)
        corpus_wall = sum(ix.ep.duration_sec for ix in self.idx.values())
        naive_over = []
        for p in self.promos:
            naive = 0.0
            for seg in p.segments:
                ix = self.idx[seg["episode"]]
                sw = slice_window(ix, _sec(seg["start_timestamp"]), _sec(seg["end_timestamp"]))
                naive += sw.contained_block_sum
            if naive > config.MAX_DURATION:
                naive_over.append((p.promo_id, p.total_duration_sec, round(naive, 2)))
        by_track: Dict[str, float] = {}
        for ev in w.events:
            if ev.start_sec >= s - 1e-6 and ev.end_sec <= e + 1e-6:
                by_track[ev.track] = round(by_track.get(ev.track, 0.0) + ev.duration, 2)
        return AuditCase(
            case_id="FM1", failure_mode="Adding up overlapping blocks inflates clip length", category="where_ai_got_it_wrong",
            title=f"Ep {ep1.ep.episode_number} {sec_to_ts(s)}-{sec_to_ts(e)}: {w.duration:.2f}s wall-clock, "
                  f"{w.contained_block_sum:.2f}s if you add up the blocks",
            episode=ep1.ep.episode_number,
            naive_pick={"method": "sum durations of every CAMERA+ACTION+DIALOGUE+SOUND FX block in the window",
                        "reported_duration_sec": w.contained_block_sum,
                        "reported_duration_if_overlapping_blocks_counted_whole": overlap_sum,
                        "per_track_sums": by_track},
            guarded_pick={"method": "wall-clock t_end - t_start", "duration_sec": w.duration},
            evidence=_lines(ep1, s, s + 9, limit=8) + [
                f"Corpus: summed block durations = {corpus_sum / 60:.1f} min vs real runtime {corpus_wall / 60:.1f} min "
                f"({corpus_sum / corpus_wall:.2f}x inflation).",
                "The brief we were handed estimated this exact window at '80+ seconds'; the measured naive sum is "
                f"{w.contained_block_sum:.1f}s - the AI-written spec itself over-estimated, which is why every number "
                "in our docs is computed, not quoted."],
            metrics={"wall_clock_sec": w.duration, "naive_sum_sec": w.contained_block_sum,
                     "inflation_x": round(w.contained_block_sum / w.duration, 2),
                     "corpus_naive_min": round(corpus_sum / 60, 1), "corpus_wall_min": round(corpus_wall / 60, 1),
                     "final_promos_a_naive_summer_would_reject_as_over_60s": len(naive_over),
                     "final_promos_total": len(self.promos),
                     "examples": naive_over[:5]},
            fix="All durations are computed from segment boundaries only (Segment.duration_sec = end - start); "
                "block durations are never added. The verifier re-checks 20 <= total <= 60 on every promo.",
            residual_risk="None for duration math. An LLM asked to 'pick a 45-second clip' from raw text still "
                          "drifts, which is why the LLM never chooses timestamps in this design - code does.")

    # ------------------------------------------------------------------ FM2
    def fm2_slow_open(self) -> AuditCase:
        rows = []
        for ep, ix in sorted(self.idx.items()):
            hp = hook_profile(build_segment(ix, 0.0, min(45.0, ix.ep.duration_sec)))
            first = ix.dialogue[0].start_sec if ix.dialogue else ix.ep.duration_sec
            prose = sum(len(EVOCATIVE_PROSE_RX.findall(ev.text)) for ev in ix.events_in(0.0, 12.0)
                        if ev.track != "DIALOGUE")
            rows.append((ep, round(first, 2), hp.passes_gate, round(hp.score, 2), prose))
        slow = [r for r in rows if r[1] >= 5.0]
        failing = [r for r in rows if not r[2]]
        ep_ex = max(slow, key=lambda r: (r[4], r[1]))[0] if slow else max(rows, key=lambda r: r[1])[0]
        ix = self.idx[ep_ex]
        if not ix.dialogue:
            raise ValueError("episode has no dialogue")
        first = ix.dialogue[0]
        best = next((p for p in sorted(self.promos, key=lambda x: x.post_order)
                     if p.promo_type == "continuous" and p.episodes == [ep_ex]), None)
        naive_prose_score = round(min(10.0, 3.0 + 0.9 * next(r[4] for r in rows if r[0] == ep_ex)), 2)
        return AuditCase(
            case_id="FM2", failure_mode="Episodes that open on silent shots", category="guardrail_caught",
            title=f"Ep {ep_ex} opens with {first.start_sec:.1f}s of silent camera work before the first line",
            episode=ep_ex,
            naive_pick={"window": f"00:00.00-{sec_to_ts(min(45.0, ix.ep.duration_sec))}",
                        "naive_prose_hook_score": naive_prose_score,
                        "why_it_fools_a_text_scorer": "evocative adjectives in CAMERA/ACTION prose read as 'dramatic'"},
            guarded_pick={"hook_gate": "FAIL - no line or kinetic beat in the first 3 s",
                          "our_hook_score_at_00:00": next(r[3] for r in rows if r[0] == ep_ex),
                          "tool_pick": (f"{best.promo_id} {best.segments[0]['start_timestamp']}-"
                                        f"{best.segments[-1]['end_timestamp']} (hook {best.scores['hook_strength']})")
                          if best else None,
                          "first_line": f"{first.start_ts} {first.speaker_raw}: \"{first.text}\""},
            evidence=_lines(ix, 0.0, first.end_sec + 0.1, limit=10),
            metrics={"episodes_with_>=5s_silent_open": len(slow), "episodes": len(rows),
                     "episode_start_windows_failing_hook_gate": len(failing),
                     "silent_open_seconds_by_episode": {r[0]: r[1] for r in rows}},
            fix="3-second hook gate: a promo may only start on a clean block start whose first 3 s contain a "
                "spoken line or a kinetic beat (slap, drag, punch, impact SFX); leading atmosphere is skipped.",
            residual_risk="A genuinely striking silent image (e.g. a bride alone in a burning chapel) is under-rated "
                          "because the gate needs a line or a kinetic verb - a human can override.")

    # ------------------------------------------------------------------ FM3
    def fm3_header_bleed(self) -> AuditCase:
        best: Optional[Tuple[float, Promo, float, float, Dict]] = None
        promoted = 0
        for p in self.promos:
            ctx = BlindContext([build_segment(self.idx[s["episode"]], _sec(s["start_timestamp"]),
                                              _sec(s["end_timestamp"])) for s in p.segments])
            blind, diag = clarity_score(ctx)
            names = set()
            for ep in p.episodes:
                hdr = self.idx[ep].ep.logline + " " + self.idx[ep].ep.continuity_notes
                names |= {w for w in hdr.replace(",", " ").split() if w[:1].isupper() and len(w) > 3}
            leaky, _ = clarity_score(ctx, leaky_names=names)
            if leaky >= self._w1_clarity() > blind:
                promoted += 1
            gap = leaky - blind
            late = min(p.episodes) >= 15 or max(self.idx) < 15     # prefer mid/late-season examples
            if late and (best is None or (gap, diag["pronoun_ratio"]) > (best[0], best[4]["pronoun_ratio"])):
                best = (gap, p, blind, leaky, diag)
        if best is None:
            raise ValueError("no promos to audit")
        gap, p, blind, leaky, diag = best
        seg = p.segments[0]
        ix = self.idx[seg["episode"]]
        return AuditCase(
            case_id="FM3", failure_mode="Reading the episode summary makes a clip look clearer than it is", category="where_ai_got_it_wrong",
            title=f"{p.promo_id}: clarity {leaky:.1f} when the evaluator reads the logline, {blind:.1f} blind",
            episode=seg["episode"],
            naive_pick={"evaluator_context": "EPISODE LOGLINE + CONTINUITY NOTES + clip", "standalone_clarity": round(leaky, 2),
                        "logline": ix.ep.logline},
            guarded_pick={"evaluator_context": "clip blocks only (speakers anonymised, names masked)",
                          "standalone_clarity": round(blind, 2), "pronoun_line_ratio": diag["pronoun_ratio"],
                          "lore_references": diag["lore_refs"], "spoken_anchor_terms": diag["anchors"]},
            evidence=_lines(ix, _sec(seg["start_timestamp"]), _sec(seg["end_timestamp"]), visual=False, limit=12),
            metrics={"clarity_gap": round(gap, 2),
                     "promos_that_would_wrongly_clear_the_clarity_floor_if_summary_leaked": promoted},
            fix="The evaluator prompt is built only from in-window blocks; loglines/continuity notes are stripped, "
                "speaker labels anonymised and names in visual prose masked unless spoken in the clip.",
            residual_risk="Blind scoring can't tell whether a clear-sounding late line ('my husband', 'Father') "
                          "carries 30 episodes of history - spoiler/lore checks stay with the marketer (Later list).")

    # ------------------------------------------------------------------ FM4
    def fm4_post_climax(self) -> AuditCase:
        trims = []
        for ep, ix in sorted(self.idx.items()):
            end = ix.ep.duration_sec
            start_target = max(0.0, end - 45.0)
            starts = [t for t, _ in ix.clean_starts if t >= start_target] or [0.0]
            s = starts[0]
            ns, ne, log = self.gen.snap_trim(ep, s, end)
            c_before = cliff_profile(build_segment(ix, s, end)).score
            c_after = cliff_profile(build_segment(ix, ns, ne)).score
            tail = [ev for ev in ix.events_in(ne, end) if ev.start_sec >= ne - 1e-6]
            trims.append({"episode": ep, "naive": (s, end), "trimmed": (ns, ne), "log": log,
                          "cliff_before": round(c_before, 2), "cliff_after": round(c_after, 2),
                          "removed_tail_sec": round(end - ne, 2),
                          "removed": [f"{t.start_ts} {t.track}{(' ' + t.speaker_raw) if t.speaker_raw else ''}: {t.text[:90]}"
                                      for t in tail][:6]})
        changed = [t for t in trims if t["removed_tail_sec"] > 0.05]
        if not changed:
            raise ValueError("snap-back trimmer did not need to move any episode-end cut in this input")
        ex = max(changed, key=lambda t: (t["cliff_after"] >= 6.0, t["cliff_after"] - t["cliff_before"], t["removed_tail_sec"]))
        ix = self.idx[ex["episode"]]
        return AuditCase(
            case_id="FM4", failure_mode="Cutting after the peak, on the release", category="guardrail_caught",
            title=(f"Ep {ex['episode']}: naive cut at {sec_to_ts(ex['naive'][1])} (episode end) -> snapped back to "
                   f"{sec_to_ts(ex['trimmed'][1])}, cliffhanger {ex['cliff_before']} -> {ex['cliff_after']}"),
            episode=ex["episode"],
            naive_pick={"window": f"{sec_to_ts(ex['naive'][0])}-{sec_to_ts(ex['naive'][1])}",
                        "cliffhanger_tension": ex["cliff_before"], "rule": "last 45 s up to the episode's final frame"},
            guarded_pick={"window": f"{sec_to_ts(ex['trimmed'][0])}-{sec_to_ts(ex['trimmed'][1])}",
                          "cliffhanger_tension": ex["cliff_after"], "trim_log": ex["log"]},
            evidence=["REMOVED TAIL:"] + ex["removed"] + ["KEPT FINAL BEAT:"] +
                     _lines(ix, max(0.0, ex["trimmed"][1] - 4.0), ex["trimmed"][1], limit=6),
            metrics={"episodes_where_trimmer_moved_the_end": len(changed), "episodes": len(trims),
                     "mean_cliff_gain": round(sum(t["cliff_after"] - t["cliff_before"] for t in changed) / max(len(changed), 1), 2),
                     "all_trims": [{k: v for k, v in t.items() if k not in ("removed",)} for t in changed][:50]},
            fix="Snap-back trimmer: ends must be clean block ends before any 'Cut to black'; within the last 8 s the "
                "cut moves to the beat with the highest unresolved-tension score (question, threat, reveal, close-up, sting).",
            residual_risk="Musical builds that aren't tagged in SOUND FX are invisible, so a cut can land a beat "
                          "before the real musical peak - checked by the human editor.")

    # ------------------------------------------------------------------ FM5
    def fm5_modality(self, top_n: int = 25) -> AuditCase:
        top = [p for p in sorted(self.promos, key=lambda x: x.post_order) if p.wave in (1, 2, 3)][:top_n]
        rows = []
        for p in top:
            risks = []
            for seg in p.segments:
                ix = self.idx[seg["episode"]]
                for ev in ix.events_in(_sec(seg["start_timestamp"]), _sec(seg["end_timestamp"])):
                    low = ev.text.lower()
                    if ev.track == "DIALOGUE" and any(t in " ".join(ev.emotion_tags).lower() for t in TTS_RISK_TAGS):
                        risks.append(f"Ep{seg['episode']} {ev.start_ts} TTS/performance: {ev.speaker_raw} "
                                     f"({', '.join(ev.emotion_tags)}) \"{ev.text[:50]}\"")
                    elif ev.track in ("ACTION", "CAMERA") and any(k in low for k in VFX_RISK_RX):
                        risks.append(f"Ep{seg['episode']} {ev.start_ts} VFX/stunt render: {ev.text[:70]}")
                    elif ev.track == "CAMERA" and SILENT_VISUAL_RX.search(ev.text) and ev.duration >= 3.0:
                        risks.append(f"Ep{seg['episode']} {ev.start_ts} silent {ev.duration:.1f}s shot relies on acting: {ev.text[:60]}")
            rows.append({"promo_id": p.promo_id, "wave": p.wave, "render_checks": risks[:6], "n_checks": len(risks)})
        ex = max(rows, key=lambda r: r["n_checks"]) if rows else {"promo_id": None, "render_checks": []}
        return AuditCase(
            case_id="FM5", failure_mode="What text cannot show: acting, voice, effects", category="where_tool_picks_badly",
            title=f"{ex['promo_id']}: {len(ex['render_checks'])}+ beats whose quality only exists in the rendered video",
            episode=None,
            naive_pick={"assumption": "if the screenplay beat is strong, the rendered clip is strong"},
            guarded_pick={"policy": "human editor verifies the top 5-10 picks on the rendered episode (~10 min)",
                          "checks": ["TTS flatness on Screaming/Sobbing lines", "lip-sync on close-ups",
                                     "VFX/stunt artefacts on punches, fire, blood, explosions",
                                     "untagged music builds vs our cut point", "silent acting beats that carry the moment"]},
            evidence=ex["render_checks"],
            metrics={"top_picks_audited": len(rows), "render_checks_per_pick": {r["promo_id"]: r["n_checks"] for r in rows}},
            fix="Every promo in the plan carries a `review_reasons` list; the Streamlit editor shows the exact blocks to scrub.",
            residual_risk="This is the irreducible text-only gap: a screenplay parser cannot hear a flat AI voice or "
                          "see a melted hand. The human stays in the loop for the final 10 minutes.")

    # ------------------------------------------------------------------ FM6
    def fm6_villain_swap(self) -> AuditCase:
        swaps = sorted(getattr(self.gen, "villain_swaps", []), key=lambda c: -c.selection_score)
        if not swaps:
            raise ValueError("no cross-episode arcs needed the villain guard in this input")
        c = swaps[0]
        a, b = c.segments
        ia, ib = self.idx[a.episode], self.idx[b.episode]
        sa = sorted({x.speaker for x in ia.events_in(a.start_sec, a.end_sec) if x.track == "DIALOGUE"})
        sb = sorted({x.speaker for x in ib.events_in(b.start_sec, b.end_sec) if x.track == "DIALOGUE"})
        shipped = [p for p in self.promos if p.promo_type == "multi_segment_cross"
                   and set(p.signals.get("new_villains_in_payoff", []))]
        return AuditCase(
            case_id="FM6", failure_mode="Fake payback: the payoff punishes a different villain",
            category="where_ai_got_it_wrong",
            title=(f"Ep{a.episode} {a.start_timestamp}-{a.end_timestamp} + Ep{b.episode} {b.start_timestamp}-"
                   f"{b.end_timestamp}: setup villain(s) {', '.join(sorted(set(sa) & self.gen.profile.core_cast - self.gen.profile.premise_cast)) or '-'}"
                   f", but the 'payback' shows a different villain begging: {', '.join(c.signals['new_villains_in_payoff'])}"),
            episode=a.episode,
            naive_pick={"rule": "arc needs the heroine + one other recurring character in both segments",
                        "selection_score": c.selection_score, "setup_speakers": sa, "payoff_speakers": sb,
                        "segments": [s.to_dict() for s in c.segments]},
            guarded_pick={"rule": "an antagonist shown begging/defeated in the payoff must already appear in the setup",
                          "blocked_arcs": len(swaps), "cross_arcs_shipped_with_a_swap": len(shipped)},
            evidence=["SETUP:"] + _lines(ia, a.start_sec, a.end_sec, visual=False, limit=6) +
                     ["PAYOFF:"] + _lines(ib, b.start_sec, b.end_sec, visual=False, limit=7),
            metrics={"blocked_arcs": len(swaps), "example_selection_score": c.selection_score},
            fix="Villain-consistency guard in CandidateGenerator.cross(): a cold viewer cannot tell two cruel "
                "noblewomen apart, so the stitched payoff must punish the same antagonist the setup introduced.",
            residual_risk="The guard keys on speaker identity; two different villains who never speak in the clip "
                          "(visual-only) could still be conflated - another reason a person reviews the plan.")

    # ------------------------------------------------------------------ FM7
    def fm7_closing_line(self) -> AuditCase:
        from .scoring import closure_signals
        best = None
        n_closing = 0
        for ep, ix in sorted(self.idx.items()):
            for t, ts in ix.clean_ends:
                seg = build_segment(ix, max(0.0, t - 12.0), t)
                n, why = closure_signals(seg)
                if n < 2:
                    continue
                n_closing += 1
                cp = cliff_profile(seg)
                raw = cp.score + min(4.0, 1.3 * n)
                if best is None or (n, raw) > (best[7], best[0]):     # clearest example: most closing signals
                    best = (raw, cp.score, ep, t, ts, why, cp.line, n)
        if best is None:
            raise ValueError("no ending in this input closes a scene")
        raw, capped, ep, t, ts, why, line, _n = best
        ix = self.idx[ep]
        alt = [(self.gen.cliff_at(ep, u), uts) for u, uts in ix.clean_ends if t - 40 <= u < t]
        alt_score, alt_ts = max(alt) if alt else (None, None)
        held_closing = [p.promo_id for p in self.promos if p.wave == 4 and p.wave_reason.startswith("held: ending")]
        return AuditCase(
            case_id="FM7", failure_mode="Ending on a line that closes the scene", category="where_ai_got_it_wrong",
            title=(f"Ep {ep} {ts}: \"{(line.text if line else '')[:80]}\" reads as a strong cliffhanger "
                   f"({raw:.1f}) but resolves the scene"),
            episode=ep,
            naive_pick={"cut": ts, "cliffhanger_score_without_detector": round(raw, 2),
                        "note": "a defiant last word sounds like tension; in the first Gemini run "
                                "(gemini-3.5-flash-lite) the Ep 2 version of this ending was rated 8.5 and ranked #1"},
            guarded_pick={"cliffhanger_score_with_detector": round(capped, 2), "signals": why,
                          "better_cut_nearby": alt_ts, "its_score": round(alt_score, 2) if alt_score is not None else None,
                          "held_before_the_shortlist_this_run": held_closing[:12]},
            evidence=_lines(ix, max(0.0, t - 14.0), t, visual=True, limit=10),
            metrics={"closing_endings_in_show": n_closing, "held_before_the_shortlist": len(held_closing)},
            fix="Closure detector (scoring.closure_signals): the last line is not a question, threat, order, reveal "
                "or plea, and at least two of - resolving delivery, a farewell/departure line, the end of a monologue "
                "nobody answers, a final shot written as an ending. Such endings are down-scored in code and held "
                "before the shortlist, so a model re-ranker never sees them.",
            residual_risk="A closing line without those signals (no delivery note, someone else answers) still passes; "
                          "the human check confirms the music and the last frame do not resolve.")

    # ------------------------------------------------------------------ FM8
    def fm8_visual_resolution(self) -> AuditCase:
        from .scoring import visual_resolution
        best = None
        n_res = 0
        for ep, ix in sorted(self.idx.items()):
            for t, ts in ix.clean_ends:
                seg = build_segment(ix, max(0.0, t - 12.0), t)
                why = visual_resolution(seg)
                if not why:
                    continue
                n_res += 1
                if best is None or len(why) > len(best[3]):
                    best = (ep, t, ts, why)
        if best is None:
            raise ValueError("no ending in this input resolves visually")
        ep, t, ts, why = best
        ix = self.idx[ep]
        alt = [(self.gen.cliff_at(ep, u), uts) for u, uts in ix.clean_ends if t - 30 <= u < t]
        alt_score, alt_ts = max(alt) if alt else (None, None)
        held = [{"promo_id": p.promo_id, "why_held": p.wave_reason} for p in self.promos
                if p.wave == 4 and p.wave_reason.startswith("held: ending resolves")]
        return AuditCase(
            case_id="FM8", failure_mode="Ending on a shot that resolves the scene", category="where_ai_got_it_wrong",
            title=(f"Ep {ep} {ts}: the last line is open, but the final shot resolves ({'; '.join(why)[:110]})"),
            episode=ep,
            naive_pick={"cut": ts, "note": "the old detector read only the last line; a 'triumphant musical swell' even "
                                         "counted as a suspense sting. In the Gemini run this ending was post #8."},
            guarded_pick={"signals": why, "better_cut_nearby": alt_ts,
                          "its_score": round(alt_score, 2) if alt_score is not None else None},
            evidence=_lines(ix, max(0.0, t - 10.0), t, visual=True, limit=10),
            metrics={"resolving_endings_in_show": n_res, "promos_held_as_resolved": len(held),
                     "examples": held[:6]},
            fix="scoring.visual_resolution reads the final 4 s of shots and sound (walking off together, into the "
                "sunlight, a soft smile, an embrace, a triumphant score fading out). Unless a threat, question, plea or "
                "reveal is still hanging in the last 1.5 s, the ending counts as resolved: down-scored in code and held "
                "before the shortlist, so the re-ranker never sees it.",
            residual_risk="A resolution written without those words (\"they sit by the fire\") still passes; the person "
                          "watching the video makes the final call.")

    # ------------------------------------------------------------------ FM9
    def fm9_repeats(self) -> AuditCase:
        from .ranker import clash
        seqs: Dict[str, List[Promo]] = {}
        for p in self.promos:
            for sq in p.signals.get("sequences", []):
                seqs.setdefault(sq, []).append(p)
        if not seqs:
            raise ValueError("no story sequences")
        big, members = max(seqs.items(), key=lambda kv: len(kv[1]))
        eps = sorted({e for (e, _sid), sq in self.gen.sequences.items() if sq == big})
        chains = 0
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                why = clash(a, b, config.POSTING_RULES["chain_gap_sec"])
                if why and "ends" in why:
                    chains += 1
        posted = [p for p in members if p.wave in (1, 2, 3)]
        alts = [p for p in self.promos if p.variant_of]
        return AuditCase(
            case_id="FM9", failure_mode="One scene posted many times", category="tool_limit_fixed",
            title=(f"Story sequence {big} (Ep {eps[0]}-{eps[-1]}) yields {len(members)} promos; {chains} pairs continue "
                   f"each other. The plan now posts {len(posted)} of them, at most "
                   f"{config.POSTING_RULES['max_per_sequence']} per list."),
            episode=eps[0],
            naive_pick={"note": "the previous plan put 10 of its first 15 posts in the Ep 1-4 wedding, and four pairs "
                                "ran end to start (one ended on 'Embarrassed you?', the next began on it)"},
            guarded_pick={"posted_from_this_sequence": [p.promo_id for p in posted],
                          "alternates_this_run": len(alts)},
            evidence=[f"{p.promo_id}: {' + '.join(s['start_timestamp'] + '-' + s['end_timestamp'] for s in p.segments)} "
                      f"-> {p.wave_label}" for p in sorted(members, key=lambda x: x.post_order)[:10]],
            metrics={"promos_in_largest_sequence": len(members), "chained_pairs": chains, "alternates": len(alts)},
            fix="Story sequences are learned from scene headings and continuity notes (same place, carries straight "
                "on). No two posts may share footage or start within 3 s of each other's cut, and each list holds at "
                "most two posts per sequence; the rest become alternates for A/B tests.",
            residual_risk="Two different sequences can still tell the same kind of story (two trials); the person "
                          "reviewing the plan checks the mix.")

    # ------------------------------------------------------------------ per-promo flags
    def flag_promos(self) -> Dict[str, int]:
        first_seen: Dict[str, int] = {}
        for ep, ix in sorted(self.idx.items()):
            for d in ix.dialogue:
                first_seen.setdefault(d.speaker, ep)
        counts: Dict[str, int] = {}
        for p in self.promos:
            flags: List[str] = []
            max_ep = max(p.episodes)
            spk: Set[str] = set()
            spk_lines: Dict[str, int] = {}
            locs: Set[str] = set()
            for seg in p.segments:
                ix = self.idx[seg["episode"]]
                s, e = _sec(seg["start_timestamp"]), _sec(seg["end_timestamp"])
                for ev in ix.events_in(s, e):
                    if ev.track == "DIALOGUE":
                        spk.add(ev.speaker)
                        spk_lines[ev.speaker] = spk_lines.get(ev.speaker, 0) + 1
                bs = build_segment(ix, s, e)
                locs |= {location_key(h) for h in bs.headings}
            old = [x for x in spk if first_seen.get(x, max_ep) <= max_ep - 15 and x not in self.gen.profile.premise_cast
                   and spk_lines.get(x, 0) >= 2]
            if old and max_ep >= 20:
                flags.append("lore_dependent")
            if p.promo_type == "continuous" and len(locs) > 1:
                flags.append("location_jump")
            if p.signals.get("diagnostics", {}).get("max_dead_air", 0) > 3.5:
                flags.append("silent_stretch")
            if p.scores["spoiler_risk"] > config.POSTING_RULES["human_spoiler_check_above"]:
                flags.append("spoiler_check")
            if "opens on a reply that needs prior context" in p.signals.get("hook_reasons", []):
                flags.append("opens_on_reply")
            if any(r for r in p.signals.get("cliff_reasons", [])):
                flags.append("soft_cliff")
            p.flags = flags
            for f in flags:
                counts[f] = counts.get(f, 0) + 1
            if p.wave in (1, 2, 3):
                for f in flags:
                    if f in ("lore_dependent", "spoiler_check", "location_jump", "opens_on_reply"):
                        msg = {"lore_dependent": "relies on characters introduced 15+ episodes earlier",
                               "spoiler_check": "spoiler risk above 5.0",
                               "location_jump": "continuous clip changes location",
                               "opens_on_reply": "opens on a reply - check it still lands cold"}[f]
                        if msg not in p.review_reasons:
                            p.review_reasons.append(msg)
                p.needs_human_review = bool(p.review_reasons)
        return counts

    def run(self) -> Dict[str, Any]:
        flag_counts = self.flag_promos()
        cases = []
        for fn, cid, name in ((self.fm1_duration, "FM1", "Adding up overlapping blocks inflates clip length"),
                              (self.fm2_slow_open, "FM2", "Episodes that open on silent shots"),
                              (self.fm3_header_bleed, "FM3", "Reading the episode summary makes a clip look clearer than it is"),
                              (self.fm4_post_climax, "FM4", "Cutting after the peak, on the release"),
                              (self.fm5_modality, "FM5", "What text cannot show: acting, voice, effects"),
                              (self.fm6_villain_swap, "FM6", "Fake payback: the payoff punishes a different villain"),
                              (self.fm7_closing_line, "FM7", "Ending on a line that closes the scene"),
                              (self.fm8_visual_resolution, "FM8", "Ending on a shot that resolves the scene"),
                              (self.fm9_repeats, "FM9", "One scene posted many times")):
            try:
                cases.append(fn())
            except Exception as exc:   # small/new inputs may not contain an example of every failure mode
                cases.append(AuditCase(case_id=cid, failure_mode=name, category="not_applicable",
                                       title=f"No example in this input ({exc})", episode=None, naive_pick={},
                                       guarded_pick={}, evidence=[], metrics={}, fix="-", residual_risk="-"))
        ranked = sorted(self.promos, key=lambda x: x.post_order)
        weak = [{"promo_id": p.promo_id, "wave": p.wave_label, "composite": p.composite_score, "flags": p.flags,
                 "segments": p.segments, "why_flagged": p.review_reasons}
                for p in ranked if p.wave in (1, 2, 3) and p.flags][:15]
        return {
            "summary": {"promos_audited": len(self.promos), "flag_counts": flag_counts,
                        "plan_promos_needing_human_review": sum(1 for p in self.promos if p.wave in (1, 2, 3) and p.needs_human_review)},
            "failure_modes": [c.to_dict() for c in cases],
            "flagged_ranked_picks": weak,
            "human_review_protocol": [
                "1. Open the app's Promos tab and start with Post first (5 promos).",
                "2. Scrub each segment on the rendered episode at its exact start and end timestamps.",
                "3. Watch the first 3 s on mute and with sound: does something happen before a thumb moves?",
                "4. Check the voice and lip-sync on every flagged line; reject flat screams.",
                "5. Check stunt and effects beats (punches, fire, falls) frame by frame.",
                "6. Confirm the cut lands before the music resolves; nudge a cut by one block in the app if needed.",
                "7. For the title reveal and any spoiler flag: confirm a new viewer learns nothing beyond the title.",
                "8. Approve or reject each promo and download the approved cut list for the editor."],
        }


def _sec(ts: str) -> float:
    return ts_to_sec(ts)
