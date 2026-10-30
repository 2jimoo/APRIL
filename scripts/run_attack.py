"""Step 2: generate poisoned passages and export misgeneration events in RAGForensics' input format.

--mode A            standard PAIR (score = attack success), then evaluate with scripts/run_forensics.py
--mode B            adaptive PAIR (RAGForensics inside the loop, score also rewards evading it)
--mode poisonedrag  no iteration: PoisonedRAG's released adv_texts, as a baseline
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pair_rag.data import load_corpus, load_prepared_targets  # noqa: E402
from pair_rag.forensics import load_ragforensics, make_trace_llm  # noqa: E402
from pair_rag.llm import make_llm, spec_to_name  # noqa: E402
from pair_rag.pair import PAIRPoisoner, feedback_path, plain_attack, to_feedback_entry  # noqa: E402
from pair_rag.rag import RAG  # noqa: E402
from pair_rag.retriever import DenseRetriever  # noqa: E402

METHOD_NAME = {"A": "PAIR", "B": "PAIR-adaptive", "poisonedrag": "PoisonedRAG"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["A", "B", "poisonedrag"], default="A")
    p.add_argument("--dataset", default="nq")
    p.add_argument("--data_root", default="data")
    p.add_argument("--retriever", default="e5")
    p.add_argument("--attacker_llm", default="ollama:qwen2.5:7b", help="claude:<model> | ollama:<model>")
    p.add_argument("--target_llm", default="ollama:qwen2.5:7b", help="LLM inside the victim RAG")
    p.add_argument("--forensics_llm", default="ollama:qwen2.5:7b", help="RAGForensics judge (mode B only)")
    p.add_argument("--attacker_effort", default="medium", help="Claude effort for the attacker")
    p.add_argument("--target_effort", default="low", help="Claude effort for the RAG generator")
    p.add_argument("--forensics_effort", default="low", help="Claude effort for the forensic judge")
    p.add_argument("--num_ctx", type=int, default=8192, help="Ollama context window")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_poison", type=int, default=5, help="M: passages injected per question")
    p.add_argument("--max_words", type=int, default=50)
    p.add_argument("--n_iterations", type=int, default=5, help="PAIR depth K")
    p.add_argument("--n_streams", type=int, default=1, help="PAIR width N (independent attacker conversations)")
    p.add_argument("--keep_last_n", type=int, default=3, help="attacker remembers the last n attempts")
    p.add_argument("--threshold", type=int, default=10, help="stop a question once SCORE >= threshold")
    p.add_argument("--no_prepend_question", action="store_true",
                   help="do not prefix each passage with the question (PoisonedRAG's S+I trick)")
    p.add_argument("--export_top_n", type=int, default=100, help="ranked texts per event (RAGForensics input)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n_questions", type=int, default=100)
    p.add_argument("--export_all", action="store_true",
                   help="also export failed attacks (RAGForensics normally only sees misgenerations)")
    p.add_argument("--feedback_root_dir", default="attack_feedback")
    p.add_argument("--log_dir", default="logs")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    data_dir = os.path.join(args.data_root, args.dataset)
    targets = load_prepared_targets(data_dir)[args.start: args.start + args.n_questions]
    retriever = DenseRetriever(args.retriever, load_corpus(data_dir), data_dir)
    target_llm = make_llm(args.target_llm, temperature=0.1, effort=args.target_effort, num_ctx=args.num_ctx)
    rag = RAG(retriever, target_llm, top_k=args.top_k)

    method = METHOD_NAME[args.mode]
    if args.mode == "poisonedrag":
        attack_llm_name = "gpt-4"  # PoisonedRAG generated adv_texts with GPT-4
        poisoner = None
    else:
        attack_llm_name = spec_to_name(args.attacker_llm)
        rf_module = rf_llm = None
        if args.mode == "B":
            rf_module = load_ragforensics()
            rf_llm = make_trace_llm(args.forensics_llm, effort=args.forensics_effort, num_ctx=args.num_ctx)
        attacker = make_llm(args.attacker_llm, temperature=1.0, top_p=0.9, effort=args.attacker_effort,
                            num_ctx=args.num_ctx)
        poisoner = PAIRPoisoner(attacker, rag, mode=args.mode, n_poison=args.n_poison, max_words=args.max_words,
                                n_iterations=args.n_iterations, n_streams=args.n_streams,
                                keep_last_n=args.keep_last_n, threshold=args.threshold,
                                prepend_question=not args.no_prepend_question, export_top_n=args.export_top_n,
                                rf_module=rf_module, rf_llm=rf_llm)

    out_path = feedback_path(args.feedback_root_dir, args.dataset, method, args.top_k, args.n_poison,
                             args.retriever, attack_llm_name)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, os.path.basename(os.path.dirname(out_path)) + "_" +
                            os.path.basename(out_path).replace(".json", f"_{int(time.time())}.jsonl"))

    events, n_success, n_retrieved_total = [], 0, 0
    with open(log_path, "w", encoding="utf-8") as log_f:
        for i, target in enumerate(targets):
            print(f"[{i + 1}/{len(targets)}] {target['question_id']}: {target['question']} -> {target['target_answer']}")
            try:
                if poisoner is None:
                    best = plain_attack(rag, target, target["poisonedrag_texts"][: args.n_poison], args.export_top_n)
                    trajectory = []
                    print(f"  success={best['success']} retrieved={best['n_retrieved']} | {best['response'][:80]!r}")
                else:
                    best, trajectory = poisoner.attack(target)
            except Exception as e:  # noqa: BLE001 - e.g. the RAG LLM is unreachable; keep going
                logging.error("question %s failed: %s", target["question_id"], e)
                best, trajectory = None, [{"error": str(e)}]
            log_f.write(json.dumps({"target": target, "trajectory": trajectory,
                                    "best": None if best is None else {k: v for k, v in best.items() if k != "ranked"}},
                                   ensure_ascii=False) + "\n")
            log_f.flush()
            if best is None:
                continue
            n_success += best["success"]
            n_retrieved_total += best["n_retrieved"]
            if best["success"] or args.export_all:
                events.append(to_feedback_entry(target, best, method))
            # write after every question so a crash keeps the finished part
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)

    n = max(1, len(targets))
    summary = {"method": method, "n_questions": len(targets), "ASR": n_success / n,
               "avg_poison_in_topk": n_retrieved_total / n, "n_events_exported": len(events),
               "feedback_file": out_path, "trajectory_log": log_path}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
