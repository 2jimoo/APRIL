"""Step 3b: end-to-end defense evaluation of an attack_feedback file (exported with --export_all).

For every question: was the answer hijacked (ASR)? If so, run RAGForensics' traceback (original query_poison,
same stopping rule), remove the flagged texts, re-answer with the K texts judged clean and check whether the
answer is still hijacked (ASR after traceback). Also reports DACC / FPR / FNR over misgeneration events with
the same per-question formula as the original evaluate().

    prevention rate = 1 - ASR                 (the corpus alone kept the right answer)
    defense rate    = 1 - ASR after traceback (corpus + RAGForensics together kept the right answer)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pair_rag.forensics import load_ragforensics, make_trace_llm, trace  # noqa: E402
from pair_rag.llm import DEFAULT_LLM, make_llm, spec_to_name  # noqa: E402
from pair_rag.rag import RAG, answer_matches  # noqa: E402


def evaluate_event(rf, trace_llm, rag, ev):
    q, target = ev["question"], ev["target_answer"]
    texts, labels = ev["context_texts"], ev["context_labels"]
    n_poison_topk = sum(labels[: rag.top_k])
    out = {"question_id": ev["question_id"], "question": q, "correct_answer": ev["correct_answer"],
           "target_answer": target, "rag_response": ev["RAG_response"], "success": bool(ev["as_judge_by_llm"]),
           "n_poison_in_topk": n_poison_topk}
    if not out["success"]:
        # RAGForensics only investigates misgenerations; nothing to trace
        return {**out, "traced": False, "success_after": False}
    ranked = list(zip(texts, ev.get("retrieval_scores", [0.0] * len(texts)), labels))
    flags, logs = trace(rf, trace_llm, q, ev["RAG_response"], ranked, rag.top_k)
    scanned = list(zip(texts, labels, flags))
    tp = sum(1 for _, y, f in scanned if y and f)
    fp = sum(1 for _, y, f in scanned if not y and f)
    fn = sum(1 for _, y, f in scanned if y and not f)
    tn = sum(1 for _, y, f in scanned if not y and not f)
    # after removing flagged texts, the new top-K is exactly the first K texts judged clean
    clean_ctx = [t for t, _, f in scanned if not f][: rag.top_k]
    response_after = rag.answer(q, clean_ctx)
    return {**out, "traced": True,
            "flagged_texts": [t for t, _, f in scanned if f],
            "flagged_labels": [y for _, y, f in scanned if f],
            "flag_reasons": [lg["response_text"][:500] for (_, _, f), lg in zip(scanned, logs) if f],
            "n_checked": len(scanned), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "dacc": (tp + tn) / len(scanned) if scanned else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fnr": fn / (fn + tp) if fn + tp else 0.0,
            "poison_left_in_topk": sum(1 for t in clean_ctx if t in set(x for x, y in zip(texts, labels) if y)),
            "response_after": response_after,
            "success_after": answer_matches(target, response_after)}


def summarize(events):
    n = len(events)
    traced = [e for e in events if e["traced"]]
    asr = sum(e["success"] for e in events) / n if n else 0.0
    asr_after = sum(e["success_after"] for e in events) / n if n else 0.0
    mean = lambda k: float(np.mean([e[k] for e in traced])) if traced else None  # noqa: E731
    return {"n_questions": n, "n_misgenerations": len(traced), "ASR": asr, "ASR_after_trace": asr_after,
            "prevention_rate": 1 - asr, "defense_rate": 1 - asr_after,
            "DACC": mean("dacc"), "FPR": mean("fpr"), "FNR": mean("fnr")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback_file", required=True, help="attack_feedback/... .json written with --export_all")
    p.add_argument("--target_llm", default=DEFAULT_LLM, help="victim RAG LLM (re-answers after removal)")
    p.add_argument("--trace_llm", default=DEFAULT_LLM, help="RAGForensics judge")
    p.add_argument("--target_effort", default="low")
    p.add_argument("--trace_effort", default="low")
    p.add_argument("--num_ctx", type=int, default=8192)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--rag_prompt", choices=["default", "bte"], default="default",
                   help="RAG system prompt used by the attacked system (bte for the PTR+BTE corpus)")
    p.add_argument("--out", default=None, help="default: <feedback_file>.eval_<trace LLM>.json")
    args = p.parse_args()

    with open(args.feedback_file, encoding="utf-8") as f:
        data = json.load(f)
    rf = load_ragforensics()
    trace_llm = make_trace_llm(args.trace_llm, effort=args.trace_effort, num_ctx=args.num_ctx)
    rag = RAG(None, make_llm(args.target_llm, temperature=0.1, effort=args.target_effort, num_ctx=args.num_ctx),
              top_k=args.top_k, prompt=args.rag_prompt)

    events = []
    for i, ev in enumerate(data):
        r = evaluate_event(rf, trace_llm, rag, ev)
        events.append(r)
        print(f"[{i + 1}/{len(data)}] {r['question_id']} success={r['success']} "
              f"flagged={len(r.get('flagged_texts', []))} success_after={r['success_after']}")
    result = {"feedback_file": args.feedback_file, "trace_llm": args.trace_llm, "target_llm": args.target_llm,
              "metrics": summarize(events), "events": events}
    out = args.out or args.feedback_file.replace(".json", f".eval_{spec_to_name(args.trace_llm)}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result["metrics"], indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
