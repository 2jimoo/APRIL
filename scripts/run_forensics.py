"""Step 3: run the ORIGINAL RAGForensics evaluate() on an attack_feedback file, with a Claude/Ollama/OpenAI judge.

Arguments mirror RAGForensics/main.py; --trace_LLM takes a provider spec instead of an OpenAI model name, and
--attack_LLM may be given as a spec too (it is only used to build the file name).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pair_rag.forensics import load_ragforensics, make_trace_llm, patch_generator  # noqa: E402
from pair_rag.llm import DEFAULT_LLM, spec_to_name  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="nq")
    p.add_argument("--attack_retriever", default="e5")
    p.add_argument("--attack_LLM", default=DEFAULT_LLM, help="attacker spec or name used in the file name")
    p.add_argument("--trace_LLM", default=DEFAULT_LLM, help="claude:<model> | ollama:<model> | openai:<model>")
    p.add_argument("--trace_effort", default="low", help="Claude effort for the forensic judge")
    p.add_argument("--attack_method", default="PAIR", help="PAIR | PAIR-adaptive | PoisonedRAG")
    p.add_argument("--attack_M", type=int, default=5)
    p.add_argument("--top_K", type=int, default=5)
    p.add_argument("--feedback_root_dir", default="attack_feedback")
    p.add_argument("--result_root_dir", default="result")
    p.add_argument("--test_version", default="v1")
    p.add_argument("--quiet", action="store_true", help="hide the per-text judge output printed by main.py")
    args = p.parse_args()

    if ":" in args.attack_LLM:
        args.attack_LLM = spec_to_name(args.attack_LLM)
    rf = load_ragforensics()
    patch_generator(rf, make_trace_llm(args.trace_LLM, effort=args.trace_effort))

    if args.quiet:
        with open(os.devnull, "w") as devnull:
            stdout, sys.stdout = sys.stdout, devnull
            try:
                result = rf.evaluate(args)
            finally:
                sys.stdout = stdout
    else:
        result = rf.evaluate(args)
    if result is None:
        sys.exit(1)
    print(json.dumps({k: result[k] for k in ("dataset", "attack_method", "DACC", "FPR", "FNR")}, indent=2))
    print(f"n_events={len(result['question_metrics'])}")


if __name__ == "__main__":
    main()
