"""Simulation: PTR+BTE built from dynamic APRIL incidents vs. PTR+BTE built from static PoisonedRAG incidents,
evaluated against unseen attack strategies.

Both corpora use the same RAGForensics defense (traceback -> remove flagged texts (PTR) -> one [LATEST] benign
text per question behind a retrieval proxy, read with the BTE prompt (BTE)). Only the incidents differ:

  bte    baseline: incidents of the static PoisonedRAG attack (one per hijacked question), paper's benign prompt
  april  ours: every hijacking attempt of APRIL with the --train_strategies is an incident; each gives a benign
         variant conditioned on it, and up to --april_max_variants variants are merged into the one entry
  naive  optional (--corpora ...,naive): original corpus, traceback + removal only

  phase 1  training attacks   APRIL x --train_strategies and PoisonedRAG, on the original corpus
  phase 2  forensics          evaluate_defense.py (RAGForensics) on the training attacks
  phase 3  PTR+BTE doc sets   build_robust_docs.py for bte and april
  phase 4  test attacks       APRIL x --test_strategies (+ PoisonedRAG), fresh against every corpus
  phase 5  evaluation         evaluate_defense.py (traceback -> removal -> re-answer) and a comparison report

Every step is skipped if its output already exists (use --overwrite to redo), so an interrupted run resumes.
"""
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pair_rag.llm import DEFAULT_LLM, spec_to_name  # noqa: E402
from pair_rag.pair import _SAFE  # noqa: E402


def run(cmd, dry):
    print("$ " + " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, check=True, cwd=ROOT)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", default="robust1", help="experiment name (output tag)")
    p.add_argument("--dataset", default="nq")
    p.add_argument("--retriever", default="e5")
    p.add_argument("--train_strategies", default="authority,temporal")
    p.add_argument("--test_strategies", default="partial,correction")
    p.add_argument("--corpora", default="bte,april", help="bte,april[,naive]; april is compared to the others")
    p.add_argument("--no_poisonedrag", action="store_true", help="skip PoisonedRAG as a test attack")
    p.add_argument("--eval_train_strategies", action="store_true",
                   help="also re-attack the defended corpora with the training strategies (in-distribution check)")
    p.add_argument("--april_benign", choices=["paper", "conditioned"], default="conditioned")
    p.add_argument("--april_max_variants", type=int, default=3)
    p.add_argument("--baseline_benign", choices=["paper", "conditioned"], default="paper")
    p.add_argument("--attacker_llm", default=DEFAULT_LLM)
    p.add_argument("--target_llm", default=DEFAULT_LLM)
    p.add_argument("--forensics_llm", default=DEFAULT_LLM, help="RAGForensics judge inside the APRIL loop")
    p.add_argument("--trace_llm", default=None, help="RAGForensics judge for evaluation (default: --forensics_llm)")
    p.add_argument("--defender_llm", default=None, help="writes the benign texts (default: --attacker_llm)")
    p.add_argument("--n_questions", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n_iterations", type=int, default=5)
    p.add_argument("--n_streams", type=int, default=1)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_poison", type=int, default=5)
    p.add_argument("--max_words", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="print the commands only")
    args = p.parse_args()
    trace_llm = args.trace_llm or args.forensics_llm
    defender_llm = args.defender_llm or args.attacker_llm
    py = sys.executable
    train = [s for s in args.train_strategies.split(",") if s]
    test = [s for s in args.test_strategies.split(",") if s]
    corpora = [c for c in args.corpora.split(",") if c]
    if "april" not in corpora or set(corpora) - {"naive", "bte", "april"}:
        p.error("--corpora must contain april and only naive, bte, april")
    others = [c for c in corpora if c != "april"]
    if set(train) & set(test):
        p.error(f"train and test strategies overlap: {set(train) & set(test)}")

    common = ["--dataset", args.dataset, "--retriever", args.retriever, "--top_k", str(args.top_k),
              "--n_poison", str(args.n_poison), "--max_words", str(args.max_words),
              "--start", str(args.start), "--n_questions", str(args.n_questions), "--export_all",
              "--target_llm", args.target_llm]
    april = ["--mode", "B", "--attacker_llm", args.attacker_llm, "--forensics_llm", args.forensics_llm,
             "--n_iterations", str(args.n_iterations), "--n_streams", str(args.n_streams)]
    defense_file = {c: os.path.join("data", args.dataset, "defense", f"{args.exp}-{c}.json") for c in ("april", "bte")}

    def attack(strategy, corpus):
        """Run one attack (strategy=None means PoisonedRAG) on one corpus; return (feedback file, eval file)."""
        tag = f"{args.exp}-{corpus}"
        if strategy is None:
            cmd = [py, "scripts/run_attack.py", "--mode", "poisonedrag", "--tag", tag] + common
            method, llm_name = f"PoisonedRAG__{tag}", "gpt-4"
        else:
            cmd = [py, "scripts/run_attack.py", "--strategy", strategy, "--tag", tag] + common + april
            method, llm_name = f"PAIR-adaptive-{strategy}__{tag}", spec_to_name(args.attacker_llm)
        if corpus != "naive":
            cmd += ["--defense_docs", defense_file[corpus]]
        # same layout as pair.feedback_path, without creating the directory (so --dry_run leaves no trace)
        fb = os.path.join("attack_feedback", args.dataset, method,
                          f"k{args.top_k}_m{args.n_poison}_{args.retriever}_{_SAFE.sub('-', llm_name)}.json")
        ev = fb.replace(".json", f".eval_{spec_to_name(trace_llm)}.json")
        if args.overwrite or not os.path.exists(os.path.join(ROOT, fb.replace(".json", ".summary.json"))):
            run(cmd, args.dry_run)
        if args.overwrite or not os.path.exists(os.path.join(ROOT, ev)):
            run([py, "scripts/evaluate_defense.py", "--feedback_file", fb, "--target_llm", args.target_llm,
                 "--trace_llm", trace_llm, "--top_k", str(args.top_k),
                 "--rag_prompt", "default" if corpus == "naive" else "bte", "--out", ev], args.dry_run)
        return fb, ev

    def trajectory_log(fb):
        path = os.path.join(ROOT, fb.replace(".json", ".summary.json"))
        if not os.path.exists(path):  # dry run
            return f"<trajectory log of {fb}>"
        with open(path, encoding="utf-8") as f:
            return json.load(f)["trajectory_log"]

    # phases 1-2: training attacks on the original corpus + RAGForensics
    train_runs = {s: attack(s, "naive") for s in train}
    static_fb, static_ev = attack(None, "naive")

    # phase 3: PTR+BTE doc sets from static (PoisonedRAG) and dynamic (APRIL) incidents
    builds = {
        "bte": ["--train_evals", static_ev, "--benign", args.baseline_benign, "--max_variants", "1"],
        "april": ["--train_logs", *[trajectory_log(fb) for fb, _ in train_runs.values()],
                  "--benign", args.april_benign, "--max_variants", str(args.april_max_variants)],
    }
    for name in ("bte", "april"):
        out = defense_file[name]
        if (name in corpora) and (args.overwrite or not os.path.exists(os.path.join(ROOT, out))):
            run([py, "scripts/build_robust_docs.py", "--defender_llm", defender_llm, "--out", out] + builds[name],
                args.dry_run)

    # phases 4-5: test attacks on every corpus
    attacks = [(s, s) for s in test] + ([] if args.no_poisonedrag else [("PoisonedRAG", None)])
    if args.eval_train_strategies:
        attacks += [(f"{s} (train)", s) for s in train]
    evals = {(name, c): attack(strategy, c)[1] for name, strategy in attacks for c in corpora}
    if args.dry_run:
        return

    covered = {}
    for name in ("bte", "april"):
        if name in corpora:
            with open(os.path.join(ROOT, defense_file[name]), encoding="utf-8") as f:
                covered[name] = set(json.load(f)["docs"])
    both = set.intersection(*covered.values())
    report = {"args": vars(args), "covered": {k: len(v) for k, v in covered.items()}, "covered_both": len(both),
              "results": []}
    for name, _ in attacks:
        runs = {}
        for corpus in corpora:
            with open(os.path.join(ROOT, evals[(name, corpus)]), encoding="utf-8") as f:
                runs[corpus] = {e["question_id"]: e for e in json.load(f)["events"]}
        for subset, qids in (("all", None), ("covered_both", both)):
            row = {"attack": name, "subset": subset}
            for corpus in corpora:
                evs = [e for q, e in runs[corpus].items() if qids is None or q in qids]
                n, traced = len(evs), [e for e in evs if e["traced"]]
                avg = lambda k: sum(e[k] for e in traced) / len(traced) if traced else None  # noqa: E731
                row[corpus] = {"n": n,
                               "ASR": sum(e["success"] for e in evs) / n if n else None,
                               "prevention_rate": 1 - sum(e["success"] for e in evs) / n if n else None,
                               "defense_rate": 1 - sum(e["success_after"] for e in evs) / n if n else None,
                               "DACC": avg("dacc"), "FPR": avg("fpr"), "FNR": avg("fnr")}
            # paired test on "defended" (not hijacked after traceback): ours vs each other corpus
            for other in others:
                qs = [q for q in runs["april"] if q in runs[other] and (qids is None or q in qids)]
                b = sum(1 for q in qs if not runs["april"][q]["success_after"] and runs[other][q]["success_after"])
                c = sum(1 for q in qs if runs["april"][q]["success_after"] and not runs[other][q]["success_after"])
                row[f"april_vs_{other}"] = {"wins": b, "losses": c, "mcnemar_p": mcnemar_exact(b, c)}
            report["results"].append(row)

    out_dir = os.path.join(ROOT, "result", args.dataset, "robust_experiment", args.exp)
    os.makedirs(out_dir, exist_ok=True)
    fmt = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    names = " / ".join(corpora)
    lines = [f"covered questions: {report['covered']}, both: {len(both)}", "",
             f"| attack | subset | defense rate {names} | prevention rate {names} | DACC {names} | FPR {names} | "
             + " | ".join(f"april vs {o} (W-L, p)" for o in others) + " |",
             "|" + "---|" * (6 + len(others))]
    for r in report["results"]:
        cell = lambda k: " / ".join(fmt(r[c][k]) for c in corpora)  # noqa: E731
        vs = " | ".join(f"{r[f'april_vs_{o}']['wins']}-{r[f'april_vs_{o}']['losses']}, "
                        f"p={r[f'april_vs_{o}']['mcnemar_p']:.3g}" for o in others)
        lines.append(f"| {r['attack']} | {r['subset']} (n={r['april']['n']}) | {cell('defense_rate')} | "
                     f"{cell('prevention_rate')} | {cell('DACC')} | {cell('FPR')} | {vs} |")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {out_dir}/report.json")


if __name__ == "__main__":
    main()
