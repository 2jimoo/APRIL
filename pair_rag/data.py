"""Download BEIR datasets + PoisonedRAG target answers and build a retrieval corpus that fits on a laptop.

The full BEIR-NQ corpus has 2.68M passages; encoding it with E5 on a Mac takes hours. We keep:
  * every passage whose Wikipedia title matches a gold passage of a target question
    (same-article passages are the hardest clean distractors, e.g. "Chicago P.D." vs "Chicago Fire")
  * a seeded random sample of the rest (--n_random)
"""
import json
import os
import random
import shutil
import subprocess
import urllib.request
import zipfile

# BEIR is mirrored on HuggingFace (BeIR/<name>, BeIR/<name>-qrels); the original TU Darmstadt zip server is
# often very slow, so it is only a fallback (--source beir).
HF_TREE = "https://huggingface.co/api/datasets/BeIR/{repo}/tree/main/{path}"
HF_FILE = "https://huggingface.co/datasets/BeIR/{repo}/resolve/main/{path}"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
POISONEDRAG_URL = "https://raw.githubusercontent.com/sleeepeer/PoisonedRAG/main/results/adv_targeted_results/{name}.json"
DEFAULT_SPLIT = {"nq": "test", "hotpotqa": "test", "msmarco": "dev"}


def _download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    # curl resumes (-C -) and shows progress; fall back to urllib when curl is missing
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", path + ".part", url], check=True)
    else:
        urllib.request.urlretrieve(url, path + ".part")
    os.replace(path + ".part", path)


def _download_hf(raw_dir, name, split):
    beir_dir = os.path.join(raw_dir, name)
    with urllib.request.urlopen(HF_TREE.format(repo=name, path="corpus")) as r:
        files = [f["path"] for f in json.load(r) if f["path"].endswith(".parquet")]
    for path in sorted(files):
        _download(HF_FILE.format(repo=name, path=path), os.path.join(beir_dir, path))
    _download(HF_FILE.format(repo=f"{name}-qrels", path=f"{split}.tsv"), os.path.join(beir_dir, "qrels", f"{split}.tsv"))
    return beir_dir


def _download_beir_zip(raw_dir, name):
    beir_dir = os.path.join(raw_dir, name)
    if not os.path.exists(os.path.join(beir_dir, "corpus.jsonl")):
        zip_path = os.path.join(raw_dir, f"{name}.zip")
        _download(BEIR_URL.format(name=name), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(raw_dir)
        os.remove(zip_path)
    return beir_dir


def download_raw(raw_dir, name, split, source="hf"):
    beir_dir = _download_hf(raw_dir, name, split) if source == "hf" else _download_beir_zip(raw_dir, name)
    _download(POISONEDRAG_URL.format(name=name), os.path.join(raw_dir, f"poisonedrag_{name}.json"))
    return beir_dir


def iter_beir_corpus(beir_dir):
    """Yield {_id, title, text} from either the BEIR corpus.jsonl or the HuggingFace parquet shards."""
    jsonl = os.path.join(beir_dir, "corpus.jsonl")
    if os.path.exists(jsonl):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                yield {"_id": str(d["_id"]), "title": d.get("title", ""), "text": d["text"]}
        return
    import pyarrow.parquet as pq

    corpus_dir = os.path.join(beir_dir, "corpus")
    for fn in sorted(os.listdir(corpus_dir)):
        if not fn.endswith(".parquet"):
            continue
        for batch in pq.ParquetFile(os.path.join(corpus_dir, fn)).iter_batches(columns=["_id", "title", "text"]):
            for _id, title, text in zip(*(batch.column(c).to_pylist() for c in ("_id", "title", "text"))):
                yield {"_id": str(_id), "title": title or "", "text": text}


def load_targets(raw_dir, name):
    """PoisonedRAG target questions: id, question, correct answer, incorrect (target) answer, adv_texts."""
    with open(os.path.join(raw_dir, f"poisonedrag_{name}.json"), encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "question_id": v["id"],
            "question": v["question"],
            "correct_answer": v["correct answer"],
            "target_answer": v["incorrect answer"],
            "poisonedrag_texts": v.get("adv_texts", []),
        }
        for v in data.values()
    ]


def load_qrels(beir_dir, split):
    qrels = {}
    with open(os.path.join(beir_dir, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def build_corpus(raw_dir, out_dir, name="nq", split=None, n_random=30000, seed=0, source="hf"):
    split = split or DEFAULT_SPLIT.get(name, "test")
    beir_dir = download_raw(raw_dir, name, split, source)
    targets = load_targets(raw_dir, name)
    qrels = load_qrels(beir_dir, split)
    gold_ids = set().union(*(qrels.get(t["question_id"], set()) for t in targets))

    # pass 1: titles of gold passages
    gold_titles = {d["title"] for d in iter_beir_corpus(beir_dir) if d["_id"] in gold_ids}
    gold_titles.discard("")

    # pass 2: keep same-title passages, reservoir-sample the rest
    rng = random.Random(seed)
    kept, reservoir, seen = [], [], 0
    for doc in iter_beir_corpus(beir_dir):
        if doc["_id"] in gold_ids or doc["title"] in gold_titles:
            kept.append(doc)
            continue
        seen += 1
        if len(reservoir) < n_random:
            reservoir.append(doc)
        else:
            j = rng.randrange(seen)
            if j < n_random:
                reservoir[j] = doc

    docs = kept + reservoir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "targets.json"), "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2, ensure_ascii=False)
    meta = {"dataset": name, "split": split, "source": source, "n_docs": len(docs), "n_gold_or_same_title": len(kept),
            "n_random": len(reservoir), "n_targets": len(targets), "seed": seed}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_corpus(data_dir):
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_prepared_targets(data_dir):
    with open(os.path.join(data_dir, "targets.json"), encoding="utf-8") as f:
        return json.load(f)
