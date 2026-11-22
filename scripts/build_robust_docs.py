"""Step 4: build a PTR+BTE doc set (RAGForensics, Appendix 8.3) from the incidents of training attacks.

An incident is one hijacked answer a user would report: the question, the wrong response and the texts
RAGForensics flagged for it (whatever their true label, as a real defender would see them). The flagged texts are
removed (PTR); then BTE adds, per question, one benign text wrapped in [LATEST]...[/LATEST] and stored behind a
retrieval proxy (the query itself), read with the BTE system prompt.

Incident sources
  --train_evals  evaluate_defense.py outputs: one incident per hijacked question (e.g. static PoisonedRAG)
  --train_logs   APRIL trajectory logs: every hijacking attempt of every iteration is an incident (dynamic attack)

Benign text
  --benign paper        the paper's generation prompt: question + correct answer only
  --benign conditioned  the paper's prompt plus the incident's wrong response and flagged texts, so each incident
                        yields a variant that answers that attack
  --max_variants V      up to V incidents per question (spread over strategies/iterations) each give one variant;
                        the variants are merged into ONE [LATEST] entry behind ONE proxy (same top-K budget)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pair_rag.llm import DEFAULT_LLM, make_llm  # noqa: E402
from pair_rag.pair import _read_prompt  # noqa: E402


def _incident(incidents, q, ev_source, question, correct, wrong, flagged):
    entry = incidents.setdefault(q, {"question": question, "correct_answer": correct, "incidents": []})
    key = (wrong, tuple(flagged))
    if all((i["wrong_response"], tuple(i["flagged_texts"])) != key for i in entry["incidents"]):
        entry["incidents"].append({"source": ev_source, "wrong_response": wrong, "flagged_texts": flagged})


def collect_incidents(eval_files=(), log_files=()):
    incidents = {}
    for path in eval_files:
        src = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as f:
            for ev in json.load(f)["events"]:
                if ev["traced"]:
                    _incident(incidents, ev["question_id"], src, ev["question"], ev["correct_answer"],
                              ev["rag_response"], ev["flagged_texts"])
    for path in log_files:
        src = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = rec["target"]
                for att in rec["trajectory"]:
                    det = att.get("detection")
                    if not att.get("success") or det is None:
                        continue
                    flagged = det.get("flagged_texts") or [att["passages"][i] for i in det["flagged_passages"]]
                    _incident(incidents, t["question_id"], f"{src}#it{att.get('iteration')}", t["question"],
                              t["correct_answer"], att["response"], flagged)
    return incidents


def spread(items, k):
    """k items evenly spread over the list (keeps variety across strategies and iterations)."""
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)] if k > 1 else items[:1]


def benign_prompt(kind, entry, incident, max_flagged):
    if kind == "paper":
        tpl = _read_prompt("bte_benign.txt")
    else:
        flagged = incident["flagged_texts"][:max_flagged]
        tpl = (_read_prompt("bte_conditioned.txt").replace("[[WRONG]]", incident["wrong_response"])
               .replace("[[FLAGGED]]", "\n".join(f"- {t}" for t in flagged) or "- (none)"))
    return tpl.replace("[[QUESTION]]", entry["question"]).replace("[[CORRECT]]", entry["correct_answer"])


def generate(llm, prompt, retries=3):
    for _ in range(retries):
        try:
            out = llm.chat([{"role": "user", "content": prompt}]).strip()
        except Exception as e:  # noqa: BLE001 - retry transient API errors
            print(f"  defender call failed: {e}")
            continue
        if out:
            return out
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_evals", nargs="*", default=[], help="evaluate_defense.py outputs")
    p.add_argument("--train_logs", nargs="*", default=[], help="APRIL trajectory logs (logs/*.jsonl)")
    p.add_argument("--benign", choices=["paper", "conditioned"], default="paper")
    p.add_argument("--max_variants", type=int, default=1, help="benign variants merged into the one [LATEST] entry")
    p.add_argument("--max_flagged", type=int, default=10, help="conditioned: flagged texts shown per incident")
    p.add_argument("--defender_llm", default=DEFAULT_LLM)
    p.add_argument("--defender_effort", default="medium")
    p.add_argument("--num_ctx", type=int, default=8192)
    p.add_argument("--same_questions_as", default=None, help="restrict to the questions of another doc set")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if not args.train_evals and not args.train_logs:
        p.error("give --train_evals and/or --train_logs")

    incidents = collect_incidents(args.train_evals, args.train_logs)
    if args.same_questions_as:
        with open(args.same_questions_as, encoding="utf-8") as f:
            keep = set(json.load(f)["docs"])
        incidents = {k: v for k, v in incidents.items() if k in keep}
    llm = make_llm(args.defender_llm, temperature=0.7, effort=args.defender_effort, num_ctx=args.num_ctx)
    docs = {}
    for i, (qid, entry) in enumerate(sorted(incidents.items())):
        chosen = spread(entry["incidents"], max(1, args.max_variants))
        variants = []
        for inc in chosen:
            text = generate(llm, benign_prompt(args.benign, entry, inc, args.max_flagged))
            if text and text not in variants:
                variants.append(text)
        if not variants:
            print(f"[{i + 1}/{len(incidents)}] {qid}: no benign text, skipped")
            continue
        docs[qid] = {**entry, "variant_sources": [c["source"] for c in chosen], "variants": variants,
                     "proxy": entry["question"], "benign_text": "[LATEST] " + " ".join(variants) + " [/LATEST]"}
        print(f"[{i + 1}/{len(incidents)}] {qid}: {len(entry['incidents'])} incidents -> {len(variants)} variants")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"defense": "PTR+BTE", "benign": args.benign, "max_variants": args.max_variants,
            "defender_llm": args.defender_llm, "train_evals": args.train_evals, "train_logs": args.train_logs,
            "n_questions_with_incidents": len(incidents),
            "n_incidents": sum(len(e["incidents"]) for e in incidents.values()), "n_covered": len(docs)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "docs": docs}, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
