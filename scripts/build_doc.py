"""Build the 2-page solution doc from the outputs of the last run, so the doc can never disagree with them.

    python scripts/build_doc.py            # writes docs/SOLUTION_DOC.md + docs/solution_doc.tex (+ .pdf if pdflatex exists)
    python scripts/build_doc.py --check    # exit 1 if docs/SOLUTION_DOC.md is out of date with raw_outputs/

Every number comes from raw_outputs/ (promos, run metadata, audit, verification, quality evaluation) and from
eval/results.json (the blind A/B test, once judged). The only hand-written facts are the assumptions, the design
choices and the AI-tools paragraph.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

OUT = config.DEFAULT_OUTPUT_DIR
DOCS = ROOT / "docs"
AUTHOR = "Gopal Krishna Tewari · B.Tech CSAM · Roll No. 2023223"


def load() -> Dict[str, Any]:
    r = lambda n: json.loads((OUT / n).read_text(encoding="utf-8")) if (OUT / n).exists() else {}
    d = r("promos_all_episodes.json")
    ev = ROOT / "eval" / "results.json"
    return {"promos": d["promos"], "meta": r("run_metadata.json"), "audit": r("quality_audit_and_bad_picks.json"),
            "verify": r("verification_report.json"), "q": r("quality_eval.json"),
            "ab": json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else {}}


def beat(text: str, n: int = 60) -> str:
    """The spoken line in quotes; a purely visual beat (no line) as plain text, without its timestamp."""
    if re.search(r'"([^"]+)"', text or ""):
        return f"“{first_quote(text, n)}”"
    t = re.sub(r"^\[(?:cut )?[\d:.]+\]\s*", "", (text or "").strip()).split(" — ")[0]
    return "(visual) " + (t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…")


def first_quote(text: str, n: int = 60) -> str:
    m = re.search(r'"([^"]+)"', text or "")
    q = (m.group(1) if m else (text or "")).strip()
    return q if len(q) <= n else q[:n].rsplit(" ", 1)[0].rstrip(".,;:!? ") + "…"


def _fm3_numbers(title: str) -> Tuple[str, str]:
    m = re.search(r"clarity ([\d.]+) when .*?([\d.]+) blind", title or "")
    return (m.group(1), m.group(2)) if m else ("?", "?")


def _segs(p: Dict[str, Any]) -> str:
    return " + ".join(f"Ep{s['episode']} {s['start_timestamp']}–{s['end_timestamp']}" for s in p["segments"]) + \
        ""


def _short_why(p: Dict[str, Any]) -> str:
    w = re.sub(r"^[^:]{3,40}: ", "", p.get("why_it_works", ""))          # drop the story-type label
    w = w.replace(" in the first second", "").replace(", and it cuts on ", "; cuts on ")   # table-width version
    return w[:1].upper() + w[1:]


def _f(x: Any, fmt: str = "{:.2f}") -> str:
    return "-" if x is None else fmt.format(x)


def blocks(D: Dict[str, Any]) -> List[Tuple]:
    P, M, A, V, Q, AB = D["promos"], D["meta"], D["audit"], D["verify"], D["q"], D.get("ab") or {}
    R = config.POSTING_RULES
    fm = {c["case_id"]: c for c in A.get("failure_modes", [])}
    plan = [p for p in sorted(P, key=lambda p: p["post_order"]) if p.get("wave") in (1, 2, 3)]
    lists = {w: [p for p in plan if p["wave"] == w] for w in (1, 2, 3)}
    first = lists[1]
    held = sum(1 for p in P if p.get("wave") == 4)
    rk = M.get("ranking", {})
    sb = M.get("show_brief", {})
    live = M.get("reranker", "keyword") != "keyword"
    bud = M.get("llm_budget", {})
    checks = V.get("checks", {})
    n_pass = sum(c["pass"] for c in checks.values())
    REP, ADS = Q.get("repetition", {}), Q.get("real_ads", {})
    prof = M.get("show_profile", {})
    lead = ", ".join(x.title() for x in prof.get("protagonists", [])) or "-"
    fm1, fm2, fm6, fm7 = (fm.get(k, {}).get("metrics", {}) for k in ("FM1", "FM2", "FM6", "FM7"))
    est = bud.get("cost_estimates_usd", {})
    lat = M.get("latency_sec", {})
    prem = next((p for p in first if p["promo_id"] == rk.get("premise_pick")), None)
    rest = [p for p in first if p is not prem]
    eps_rest = sorted({s["episode"] for p in rest for s in p["segments"]})
    finale = rk.get("finale_first_episode")
    stage2 = f"the model {M['reranker'].split(':')[-1]}" if live else "the free keyword re-ranker"

    B: List[Tuple] = []
    B.append(("title", "Find the Promos: picking microdrama ads from the screenplay"))
    B.append(("sub", AUTHOR))
    B.append(("small", f"Kuku AI Generalist Assignment, Problem 1. Show: *{M['show_title']}*, {M['episodes_parsed']} "
                       f"episodes, {M['corpus_runtime_min']:.0f} min. Re-ranked by {stage2}; every number is generated "
                       f"from `raw_outputs/` and `eval/`."))
    # 1 ----------------------------------------------------------------------------------------------------------
    B.append(("h", "1. Why this problem, and my assumptions"))
    B.append(("p", "I chose Problem 1 because it is a real weekly cost with a checkable answer. Each new show is "
                   f"about {M['corpus_runtime_min']:.0f} minutes of footage an editor watches before picking anything, "
                   "and the promo is the ad a new viewer sees before installing, so a better first 3 seconds lowers "
                   "the cost of each install. Unlike Problems 2 and 3, it has real input and an output an editor can "
                   "use the same day (an episode, a start, an end). Where the brief was silent I assumed:"))
    B.append(("ul", [
        f"**Hook** = a line or a physical action in the first 3 s; the clip starts {M.get('pre_roll_sec', 0.3)} s "
        f"before it ({fm2.get('episodes_with_>=5s_silent_open', '?')} of {fm2.get('episodes', '?')} episodes open on "
        "5 s+ of silent shots).",
        "**Cliffhanger** = the cut lands on an open beat (question, threat, order, reveal, plea) and the last shots "
        "and sound do not settle it. **Makes sense alone** is judged blind, without the episode summary.",
        f"**Length** = end − start (the tracks overlap: summing blocks is {fm1.get('inflation_x', '?')}× too long on "
        "Ep 1); the screenplay's timestamps match the edited episode; English only.",
        "**The twist the title promises is a hook, not a spoiler; the finale is.** "
        + (f"Ep {finale}–{M['episodes_parsed']} (the last 10%) never go to a cold audience, only to viewers already deep "
           "in the show." if finale else "")]))
    # 2 ----------------------------------------------------------------------------------------------------------
    B.append(("h", "2. How it works: code shortlists, a re-ranker orders"))
    B.append(("ul", [
        f"**Stage 1, code (free, same answer every run).** It learns the show from the screenplays and the title: the "
        f"lead ({lead}), the premise (*{sb.get('premise_setup', '-')}* → *{sb.get('premise_twist', '-')}*) and this "
        f"show's own stakes words ({', '.join(sb.get('stakes_terms', [])[:4]) or '-'}), learned from its threat and "
        "reveal lines. It tries "
        f"{M.get('windows_evaluated', 0):,} windows, cuts {M.get('candidates', M['promos'])} clips (stretches, "
        "jump-cuts, cross-episode arcs, setup + twist montages), rates each blind and holds those that fail a test; "
        f"the best {M.get('shortlisted', '?')} pass to stage 2.",
        "**Stage 2, a swappable re-ranker** orders the shortlist and writes the *why it works* line: free keyword "
        "rules by default, or with one flag Gemini, OpenAI (any compatible endpoint) or Anthropic, scoring clips "
        f"0–100 in groups of {config.RERANK_GROUP_SIZE}. Code rejects a line quoting words the clip never says; answers "
        "are cached and a failing group keeps the keyword order, so a live demo never stalls.",
        f"**Code builds the plan:** hold a clip that fails a test (hook ≥ {R['min_hook']:.0f}/10, open ending, makes "
        f"sense alone ≥ {R['min_clarity']:.0f}/10); *Post first* = {R['post_first']} (one slot for the clip that sells "
        f"the title's twist, the rest from the first quarter, spoiler ≤ {R['first_max_spoiler']}); *Post next* = "
        f"{R['post_next']}; *Later* ≤ {R['later']}, the only list for finale clips; no shared footage, at most "
        f"{R['max_per_sequence']} per story sequence; order = score.",
        "**A person approves** each pick in the app (about 10 min per show) and exports the list."]))
    # 3 ----------------------------------------------------------------------------------------------------------
    B.append(("h", "3. What to post first, and why"))
    B.append(("p", f"{len(plan)} promos are in the plan ({len(first)} Post first, {len(lists[2])} Post next, "
                   f"{len(lists[3])} Later) and {held} are held, each with its reason. Post first, in posting order:"))
    rows = [[str(p["post_order"]), _segs(p), f"{beat(p['hook'], 44)} → {beat(p['cliffhanger'], 44)}", _short_why(p)]
            for p in first]
    B.append(("table", ["#", "Segments", "Hook line → cut line", "Why it works"], rows,
              r"@{}r >{\raggedright\arraybackslash}p{4.1cm} Y Y@{}"))
    smax = max((p["scores"]["spoiler_risk"] for p in rest), default=0)
    why_first = ("**Why these first:** each opens on a confrontation within a second and cuts on an unanswered "
                 "threat, plea or question. ")
    if prem:
        pe = [s["episode"] for s in prem["segments"]]
        why_first += (f"#{prem['post_order']} sells the premise: a setup from Ep {pe[0]} joined to the line that "
                      f"lands the twist in Ep {pe[-1]}, which the title already promises. ")
    if eps_rest:
        why_first += (f"The others come from Ep {eps_rest[0]}–{eps_rest[-1]}, so a click lands near the start, and "
                      f"none gives away a later twist (spoiler ≤ {smax:.1f}/10). ")
    why_first += "All lists: `raw_outputs/ranked_promos.csv`."
    B.append(("p", why_first))
    # 4 ----------------------------------------------------------------------------------------------------------
    B.append(("h", "4. How I know it is good"))
    judges = AB.get("judges", {})
    B.append(("p", "**Blind A/B test.** 30 clips, shuffled, with the source hidden: the 15 promos of Post first and "
                   "Post next against 15 baselines, the last 40 s of a random non-finale episode (what an editor grabs "
                   "without reading, since episodes end on a cliffhanger). Judges score hook, ending, makes sense "
                   "alone (1–5) and would-post (1, ½, 0); win rate = share of plan × baseline pairs the plan wins."))
    if judges:
        jrows = []
        for name, s in list(judges.items()) + ([("All judges", AB["pooled"])] if AB.get("pooled") and len(judges) > 1 else []):
            pl, bl = s["pipeline"], s["baseline"]
            jrows.append([_judge_name(name)] + [f"{_f(pl[k])} vs {_f(bl[k])}" for k in ("hook", "ending", "clarity", "would_post")]
                         + [_f(s.get("win_rate"))])
        B.append(("table", ["Judge", "Hook", "Ending", "Makes sense", "Would post", "Win rate"], jrows,
                  r"@{}Y r r r r r@{}"))
        ag = AB.get("agreement", [])
        if ag:
            B.append(("p", "Plan vs baseline in each cell. Rank agreement between judges: "
                           + ", ".join(f"{_f(a['total_spearman'])}" for a in ag[:3])
                           + " (`eval/results.md`)."))
    else:
        B.append(("p", "*Results pending:* the packet is built (`eval/packet.md`); run "
                       "`python scripts/blind_eval.py judge`, add a second model's answers as "
                       "`eval/judge2_<model>.json` and my labels in `eval/manual_labels.csv`, then `merge`."))
    sa = Q.get("scorer_agreement", {})
    if sa.get("available"):
        B.append(("p", f"**Model vs code.** On the shortlist the model's order agrees with the stage-1 score at "
                       f"{sa['composite_spearman']} (rank correlation); {sa['top10_overlap']} of the top 10 are the same."))
    if ADS.get("plan"):
        row = {r["pattern"]: r for r in ADS["rows"]}
        durs = sorted(p["total_duration_sec"] for p in plan)
        B.append(("p", "**Also:** top ReelShort and DramaBox ads (AdMapix, 2026) open on a conflict by 2 s and run "
                       "12–64 s; here " + row["Open on a line or a visible threat"]["this_plan"].split(" and all")[0]
                       + f" and posts run {durs[0]:.0f}–{durs[-1]:.0f} s. {n_pass}/{len(checks)} automatic checks pass "
                       "(timestamps verbatim, 20–60 s, no repeats, no finale in cold lists)."))
    # 5 ----------------------------------------------------------------------------------------------------------
    B.append(("h", "5. Where it picks badly"))
    bad = []
    per = AB.get("per_clip", [])
    names = list(judges)
    if per and names:
        def tot(c):
            rs = [c[n] for n in names if c.get(n)]
            return sum(r["hook"] + r["ending"] + r["clarity"] for r in rs) / len(rs) if rs else 0
        worst = sorted([c for c in per if c["source"] == "pipeline"], key=tot)[:2]
        if worst:
            bad.append("**Weakest posts** by the judges: " + "; ".join(
                " + ".join("Ep%d %s" % (s["episode"], s["start_timestamp"]) for s in c["segments"]) for c in worst)
                + f"; {sum(1 for c in per if c['source'] == 'baseline' and tot(c) >= tot(worst[0]))} baselines "
                  "score at least as well.")
    bad.append(f"**Too strict on clarity:** {rk.get('held_unclear', 0)} clips held as unclear; it counts names and "
               "pronouns, not stakes.")
    top15 = REP.get("this_run", {}).get("Top 15", {})
    if top15:
        bad.append(f"**One sequence leads:** {top15['most_from_one_sequence']} of the first 15 are from "
                   f"{_seq_eps(REP, top15.get('that_sequence'))} (two per list).")
    bad.append("**Text cannot show acting or effects**, so a person watches the video before posting.")
    bad.append("**Where AI got it wrong, 1: it fit one show.** My AI-assisted first version typed this show's own "
               "plot words and title noun into the scoring. Now the premise comes from the title and the stakes words "
               "from each script's threat lines; a test fails if a learned word appears in the code.")
    bad.append("**Where AI got it wrong, 2:** Gemini, as an early scorer, ranked Ep 2's dignified exit #1 though the "
               f"scene resolves; a closing-line check now catches {fm7.get('closing_endings_in_show', '?')} such "
               "endings. My clip joiner cut from one villain's cruelty to a different villain begging; "
               f"{fm6.get('blocked_arcs', '?')} such arcs are now blocked.")
    B.append(("ul", bad))
    # 6 ----------------------------------------------------------------------------------------------------------
    B.append(("h", "6. How it scales to every show we make"))
    per_show = est.get("gemini-3.5-flash-lite", 0)
    j2 = next((re.sub(r"^Judge 2[.\d]* \((.+)\)$", r"\1", n) for n in judges if n.startswith("Judge 2")), "")
    run_min, secs = M["corpus_runtime_min"], lat.get("total", 0)
    B.append(("ul", [
        f"**Per show:** about {secs:.0f} s and $0 with the keyword re-ranker (with a model: {bud.get('calls', '?')} "
        f"calls, ≈${per_show:.2f}; repeats are cached). The editor reviews {len(plan)} clips in about 10 minutes "
        f"instead of watching {run_min:.0f} min.",
        f"**Every week:** 10 new shows ≈ {10 * secs / 60:.0f} min of compute, under ${max(1, round(10 * per_show + 0.5))} "
        f"with a model, about {10 * 10 / 60:.0f} h of review instead of {10 * run_min / 60:.0f} h of watching; one "
        "command per show folder, so a nightly job can run each new show.",
        "**A new show needs no setup:** cast, lead, premise and stakes words are learned from its own screenplays and "
        "title (a 3-episode show in another screenplay format passes every test).",
        "**Other languages and formats:** the rules and statistics are language-neutral, but the hook and threat "
        "word lists are English (translate them, or use the model path); a new screenplay layout needs a small "
        "parser change. Both untested.",
        "**Better over time:** the app saves every approve and reject; joined with each posted promo's 3-second hold "
        "rate and cost per install, they become labels to re-fit the score weights each month.",
        "**AI tools:** Claude (coding assistant), Gemini (optional re-ranker, Judge 1)"
        + (f", {j2} (Judge 2)." if j2 else ".")]))
    return B


def _judge_name(name: str) -> str:
    """'Judge 1 (gemini:gemini-2.5-flash)' -> 'Judge 1: gemini-2.5-flash'."""
    m = re.match(r"(.+?) \((?:[a-z]+:)?(.+)\)$", name)
    return f"{m.group(1)}: {m.group(2)}" if m and not name.startswith("Person") else name.replace(" (manual labels)", "")


def _dash(clip: str) -> str:
    return re.sub(r"(\d)-(\d)", r"\1–\2", clip)


def _seq_eps(REP: Dict[str, Any], sid: Any) -> str:
    eps = REP.get("sequence_episodes", {}).get(sid or "", [])
    return (f"Ep {eps[0]}–{eps[-1]}" if eps[-1] != eps[0] else f"Ep {eps[0]}") if eps else "one sequence"


def _fm8_shot(title: str) -> str:
    m = re.search(r"final shot resolves: [\u201c\"]([^\u201d\"]+)", title or "")
    return m.group(1) if m else "the final shot"


# ------------------------------------------------------------------------------------------- renderers
def to_md(B: List[Tuple]) -> str:
    L: List[str] = []
    for b in B:
        k = b[0]
        if k == "title":
            L.append(f"# {b[1]}\n")
        elif k == "sub":
            L.append(f"{b[1]}\n")
        elif k == "small":
            L.append(f"{b[1]}\n")
        elif k == "h":
            L.append(f"## {b[1]}\n")
        elif k == "p":
            L.append(b[1] + "\n")
        elif k == "ul":
            L.extend(f"- {x}" for x in b[1])
            L.append("")
        elif k == "table":
            L.append("| " + " | ".join(b[1]) + " |")
            L.append("|" + "---|" * len(b[1]))
            L.extend("| " + " | ".join(r) + " |" for r in b[2])
            L.append("")
    return "\n".join(L).rstrip() + "\n"


TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_inline(t: str) -> str:
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", t)
    out = []
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            out.append(r"\textbf{" + tex_inline(part[2:-2]) + "}")
        elif part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + "".join(TEX_ESC.get(c, c) for c in part[1:-1]) + "}")
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(r"\emph{" + tex_inline(part[1:-1]) + "}")
        else:
            s = "".join(TEX_ESC.get(c, c) for c in part)
            s = (s.replace("−", "--").replace("–", "--").replace("—", "---").replace("…", r"\ldots{}")
                 .replace("≈", r"$\approx$").replace("×", r"$\times$").replace("→", r"$\rightarrow$")
                 .replace("κ", r"$\kappa$").replace("“", "``").replace("”", "''").replace("·", r"\textperiodcentered{}")
                 .replace("’", "'").replace("≤", r"$\le$").replace("≥", r"$\ge$")
                 .replace("½", r"\textonehalf{}"))
            out.append(s)
    return "".join(out)


def to_tex(B: List[Tuple]) -> str:
    L = [r"\documentclass[11pt]{article}",
         r"\usepackage[a4paper,margin=1.8cm]{geometry}",
         r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage{lmodern}",
         r"\usepackage{textcomp}", r"\usepackage{tabularx,booktabs,array,xltabular}", r"\usepackage{enumitem}",
         r"\usepackage[hidelinks]{hyperref}",
         r"\hypersetup{pdftitle={Find the Promos},pdfauthor={Gopal Krishna Tewari}}", r"\usepackage{titlesec}",
         r"\titleformat{\section}{\large\bfseries}{}{0pt}{}", r"\titlespacing*{\section}{0pt}{7pt}{3pt}",
         r"\setlist{nosep,leftmargin=14pt,topsep=2pt}", r"\setlength{\parindent}{0pt}", r"\setlength{\parskip}{4pt}",
         r"\newcolumntype{Y}{>{\raggedright\arraybackslash}X}", r"\pagestyle{empty}", r"\begin{document}"]
    for b in B:
        k = b[0]
        if k == "title":
            L.append(r"{\Large\bfseries " + tex_inline(b[1]) + r"}\\[2pt]")
        elif k == "sub":
            L.append(tex_inline(b[1]) + r"\\[1pt]")
        elif k == "small":
            L.append(r"{\small\raggedright " + tex_inline(b[1]) + r"\par}" + "\n")
        elif k == "h":
            L.append(r"\section*{" + tex_inline(b[1]) + "}")
        elif k == "p":
            L.append(tex_inline(b[1]) + "\n")
        elif k == "ul":
            L.append(r"\begin{itemize}")
            L.extend(r"\item " + tex_inline(x) for x in b[1])
            L.append(r"\end{itemize}")
        elif k == "table":
            spec = b[3] if len(b) > 3 else r"@{}" + "l" * len(b[1]) + "@{}"
            L.append(r"{\small\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}"
                     r"\begin{xltabular}{\linewidth}{" + spec + r"}\toprule")
            L.append(" & ".join(r"\textbf{" + tex_inline(h) + "}" for h in b[1]) + r"\\\midrule\endhead")
            L.extend(" & ".join(tex_inline(c) for c in r) + r"\\" for r in b[2])
            L.append(r"\bottomrule\end{xltabular}}" + "\n")
    L.append(r"\end{document}")
    return "\n".join(L) + "\n"


def pdf_pages(pdf: Path) -> int:
    """Page count from pdflatex's own log ("Output written on x.pdf (2 pages, ...)")."""
    log = pdf.with_suffix(".log").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if docs/SOLUTION_DOC.md is stale")
    a = ap.parse_args()
    B = blocks(load())
    md = to_md(B)
    if a.check:
        cur = (DOCS / "SOLUTION_DOC.md").read_text(encoding="utf-8") if (DOCS / "SOLUTION_DOC.md").exists() else ""
        if cur != md:
            print("docs/SOLUTION_DOC.md is out of date with raw_outputs/ - run: python scripts/build_doc.py")
            return 1
        print("docs match raw_outputs/")
        return 0
    DOCS.mkdir(exist_ok=True)
    (DOCS / "SOLUTION_DOC.md").write_text(md, encoding="utf-8")
    (DOCS / "solution_doc.tex").write_text(to_tex(B), encoding="utf-8")
    words = len(re.findall(r"\w+", md))
    print(f"wrote docs/SOLUTION_DOC.md and docs/solution_doc.tex ({words} words)")
    if shutil.which("pdflatex"):
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "solution_doc.tex"], cwd=DOCS,
                               capture_output=True, text=True)
        pdf = DOCS / "solution_doc.pdf"
        if r.returncode != 0 or not pdf.exists():
            print(r.stdout[-2000:])
            return 1
        n = pdf_pages(pdf)
        print(f"wrote docs/solution_doc.pdf ({n} page{'s' if n != 1 else ''})")
        if n > 2:
            print("ERROR: the brief allows 2 pages - cut words (do not shrink the type)")
            return 1
    else:
        print("pdflatex not found: compile docs/solution_doc.tex on Overleaf, or install MiKTeX / TeX Live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
