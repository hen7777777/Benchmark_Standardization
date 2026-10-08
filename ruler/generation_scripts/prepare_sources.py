"""Freeze RULER inputs locally; later generation never downloads data."""
import argparse
import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path

import html2text
import nltk
import requests
from bs4 import BeautifulSoup

SCRIPTS_DIR = Path(__file__).resolve().parent
SUITE_ROOT = SCRIPTS_DIR.parents[1]  # harness_builtin_suite
DEFAULT_SOURCES = SUITE_ROOT / "ruler" / "sources"
DEFAULT_RULER_REPO = SUITE_ROOT / "vendor" / "RULER"


def download(url, dest):
    if dest.exists():
        return
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--ruler-repo", type=Path, default=DEFAULT_RULER_REPO,
                        help="RULER source repo; only needed for cold re-download of sources")
    opts = parser.parse_args()
    sources = opts.sources_dir.resolve()
    ruler_repo = opts.ruler_repo.resolve()
    tokenizer_rev = "c1899de289a04d12100db370d81485cdf75e47ca"
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]:
        download(f"https://huggingface.co/Qwen/Qwen3-0.6B/resolve/{tokenizer_rev}/{name}", sources / "tokenizer" / name)
    (sources / "tokenizer_source.json").write_text(json.dumps({"repo_id": "Qwen/Qwen3-0.6B", "revision": tokenizer_rev}, indent=2) + "\n")
    generator = sources / "generator"
    if not generator.exists():
        shutil.copytree(ruler_repo / "scripts/data", generator)
    shutil.copyfile(ruler_repo / "scripts/synthetic.yaml", sources / "task_configs.yaml")
    json_dir = generator / "synthetic/json"
    ruler_rev = subprocess.check_output(["git", "-C", str(ruler_repo), "rev-parse", "HEAD"], text=True).strip()
    words_file = json_dir / "english_words.json"
    words_url = f"https://media.githubusercontent.com/media/NVIDIA/RULER/{ruler_rev}/scripts/data/synthetic/json/english_words.json"
    if words_file.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
        words_file.unlink()
    download(words_url, words_file)
    qa_urls = {
        "squad.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "hotpotqa.json": "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
    }
    for name, url in qa_urls.items():
        download(url, json_dir / name)
    urls = (json_dir / "PaulGrahamEssays_URLs.txt").read_text().splitlines()

    def essay(url):
        effective = url.replace("http://", "https://")
        effective = effective.replace("https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/", "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/")
        kind = "html" if ".html" in url else "repo"
        raw = sources / "essays_raw" / kind / url.rsplit("/", 1)[-1]
        download(effective, raw)
        encoding = "utf-8"
        try:
            text = raw.read_bytes().decode(encoding)
        except UnicodeDecodeError:
            # Paul Graham's legacy pages contain actual Windows-1252 punctuation.
            encoding = "windows-1252"
            text = raw.read_bytes().decode(encoding)
        if kind == "html":
            h = html2text.HTML2Text()
            h.ignore_images = h.ignore_tables = h.escape_all = True
            h.reference_links = h.mark_code = False
            tag = BeautifulSoup(text, "html.parser").find("font")
            if tag is None:
                raise ValueError(f"No essay font element: {url}")
            text = h.handle(str(tag))
        dest = sources / "essays_text" / kind / raw.name.replace(".html", ".txt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return {"url": url, "effective_url": effective, "encoding": encoding, "text_file": str(dest.relative_to(SUITE_ROOT))}

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        provenance = list(executor.map(essay, urls))
    ordered = sorted((sources / "essays_text/repo").glob("*.txt")) + sorted((sources / "essays_text/html").glob("*.txt"))
    text = "".join(p.read_text() for p in ordered)
    (json_dir / "PaulGrahamEssays.json").write_text(json.dumps({"text": text}))
    (sources / "corpus_sources.json").write_text(json.dumps({"qa": qa_urls, "english_words": words_url, "essays": provenance, "essay_concat_order": [str(p.relative_to(SUITE_ROOT)) for p in ordered], "decoding": "UTF-8, Windows-1252 for legacy pages; encoding recorded per URL", "all_essays_required": True}, indent=2) + "\n")
    nltk.download("punkt", download_dir=str(sources / "nltk_data"), raise_on_error=True)
    nltk.download("punkt_tab", download_dir=str(sources / "nltk_data"), raise_on_error=True)
    print(f"Frozen {len(provenance)} essays, SQuAD and HotpotQA", flush=True)


if __name__ == "__main__":
    main()
