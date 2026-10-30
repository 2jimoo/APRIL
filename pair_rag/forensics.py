"""Adapter around the unmodified RAGForensics code in third_party/RAGForensics.

The original `main.py` hard-codes an OpenAI client (`OpenAIGenerator`). Our LLM classes implement the same
`generate(list_of_messages) -> list[str]` contract, so we swap the class in and reuse the original
`judge_content_by_incorrect_answer` prompt, `query_poison` and `evaluate` verbatim.
"""
import contextlib
import importlib.util
import io
import os
import sys

from .llm import make_llm

_RF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "RAGForensics")


def load_ragforensics():
    if "ragforensics_main" in sys.modules:
        return sys.modules["ragforensics_main"]
    sys.path.insert(0, _RF_DIR)  # main.py does `from OpenAI_API import OpenAIGenerator`
    spec = importlib.util.spec_from_file_location("ragforensics_main", os.path.join(_RF_DIR, "main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["ragforensics_main"] = module
    spec.loader.exec_module(module)
    return module


def make_trace_llm(spec, effort="low", num_ctx=8192):
    # RAGForensics' own config: temperature=0.1, top_p=0.1 (ignored for Claude, which rejects sampling params)
    return make_llm(spec, temperature=0.1, top_p=0.1, effort=effort, num_ctx=num_ctx)


def patch_generator(module, llm):
    """Make the original evaluate() build our LLM instead of OpenAIGenerator."""
    module.OpenAIGenerator = lambda config: llm


def trace(module, llm, question, rag_response, ranked, top_k, verbose=False):
    """Run RAGForensics' iterative traceback on one misgeneration event.

    Same loop as main.py: walk the retrieval ranking, ask the LLM about each text, stop once top_k texts
    were judged clean. (Removing flagged texts and re-retrieving top-K until no poison is left is equivalent
    to this single sequential scan, because clean-judged texts stay in the window.)
    """
    flags, logs, clean = [], [], 0
    for text, _, is_poison in ranked:
        buf = io.StringIO()
        with contextlib.redirect_stdout(sys.stdout if verbose else buf):
            flagged, res = module.query_poison(llm, question, rag_response, text, is_poison)
        flags.append(flagged)
        logs.append(res[0])
        if not flagged:
            clean += 1
            if clean >= top_k:
                break
    return flags, logs
