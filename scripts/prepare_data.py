"""Step 1: download BEIR + PoisonedRAG targets, build the laptop-sized corpus, pre-compute embeddings."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pair_rag.data import build_corpus, load_corpus  # noqa: E402
from pair_rag.retriever import DenseRetriever  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="nq", help="BEIR dataset name (nq | hotpotqa | msmarco)")
    p.add_argument("--split", default=None, help="qrels split (default: test, dev for msmarco)")
    p.add_argument("--n_random", type=int, default=30000, help="random clean passages added to the corpus")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", choices=["hf", "beir"], default="hf", help="HuggingFace mirror (fast) or BEIR zip")
    p.add_argument("--retriever", default="e5")
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--data_root", default="data")
    args = p.parse_args()

    out_dir = os.path.join(args.data_root, args.dataset)
    meta = build_corpus(args.raw_dir, out_dir, args.dataset, args.split, args.n_random, args.seed, args.source)
    print(meta)
    DenseRetriever(args.retriever, load_corpus(out_dir), out_dir)
    print(f"embeddings cached in {out_dir}/emb_{args.retriever}.npy")


if __name__ == "__main__":
    main()
