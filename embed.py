"""Phase 3 — turn Phase 2 symbols into searchable vectors in ChromaDB.

Chunking is symbol-aligned: one chunk per function/method/class, never an
arbitrary character window that slices a function in half. Oversized functions
are split into overlapping line windows so nothing is silently truncated by the
embedding model's input limit.

Run:
    python embed.py <repo-url>                    # build the index
    python embed.py <repo-url> -q "how requests are routed"
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from analyze import Analysis, Symbol, analyze
from ingest import REPOS_DIR

# Small + fast: 384 dims, runs fine on CPU. Torch here is CPU-only.
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = Path(__file__).parent / ".chroma"

# MiniLM truncates at 256 word-pieces (~1000 chars of code). Split above this so
# the tail of a long function still gets embedded instead of being dropped.
MAX_CHUNK_CHARS = 1000
OVERLAP_LINES = 8

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Loaded once — first call downloads ~90MB from HuggingFace."""
    global _model
    if _model is None:
        print(f"[embed] loading {MODEL_NAME} (first run downloads it)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@dataclass
class Chunk:
    chunk_id: str
    text: str  # what actually gets embedded
    qualname: str
    kind: str
    file: str
    start_line: int
    end_line: int
    part: int
    n_parts: int


def _split_long(source: str, limit: int = MAX_CHUNK_CHARS) -> list[tuple[str, int]]:
    """Split on line boundaries into overlapping windows. Returns (text, line_offset)."""
    lines = source.splitlines()
    if len(source) <= limit:
        return [(source, 0)]

    windows: list[tuple[str, int]] = []
    start = 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and size + len(lines[end]) + 1 <= limit:
            size += len(lines[end]) + 1
            end += 1
        if end == start:  # single line longer than the limit
            end = start + 1
        windows.append(("\n".join(lines[start:end]), start))
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)
    return windows


def _header(sym: Symbol, analysis: Analysis) -> str:
    """Plain-English context prepended to the code.

    The embedding model was trained on prose, not on Python. Naming the symbol,
    its file, and its neighbours in the call graph gives it far more to match a
    natural-language query against than raw syntax alone.
    """
    bits = [f"{sym.kind} {sym.qualname}", f"defined in {sym.file}"]
    if sym.owner_class:
        bits.append(f"method of class {sym.owner_class.split('.')[-1]}")
    if sym.docstring:
        bits.append(f"documentation: {sym.docstring[:300]}")
    callees = [c.split(".")[-1] for c in analysis.callees(sym.qualname)[:8]]
    if callees:
        bits.append("calls: " + ", ".join(callees))
    callers = [c.split(".")[-1] for c in analysis.callers(sym.qualname)[:5]]
    if callers:
        bits.append("called by: " + ", ".join(callers))
    return "\n".join(bits)


def _class_summary(sym: Symbol) -> str:
    """Classes embed as signature + docstring only.

    Embedding a whole class body would duplicate every method it contains and
    swamp the index with near-identical vectors.
    """
    head = sym.source.split("\n")
    keep: list[str] = []
    for line in head:
        keep.append(line)
        if line.rstrip().endswith(":") and "class " in line:
            break
    return "\n".join(keep[:5])


def build_chunks(analysis: Analysis) -> list[Chunk]:
    chunks: list[Chunk] = []
    for qual, sym in sorted(analysis.symbols.items()):
        body = _class_summary(sym) if sym.kind == "class" else sym.source
        windows = _split_long(body)
        for i, (text, line_off) in enumerate(windows):
            chunks.append(Chunk(
                chunk_id=f"{qual}#{i}",
                text=f"{_header(sym, analysis)}\n\n{text}",
                qualname=qual,
                kind=sym.kind,
                file=sym.file,
                start_line=sym.start_line + line_off,
                end_line=min(sym.start_line + line_off + text.count("\n"), sym.end_line),
                part=i + 1,
                n_parts=len(windows),
            ))
    return chunks


def collection_name(root: Path) -> str:
    """Chroma requires 3-512 chars, alphanumeric/_/- , alphanumeric at both ends."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", root.name).strip("_-")
    return f"repo_{slug}"[:500]


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def build_index(analysis: Analysis, rebuild: bool = True) -> chromadb.Collection:
    chunks = build_chunks(analysis)
    name = collection_name(analysis.root)
    client = _client()

    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # not there yet — fine

    col = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine pairs with normalized vectors
    )

    print(f"[embed] {len(chunks)} chunks from {len(analysis.symbols)} symbols")
    vectors = get_model().encode(
        [c.text for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    col.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=[v.tolist() for v in vectors],
        documents=[c.text for c in chunks],
        metadatas=[{
            "qualname": c.qualname, "kind": c.kind, "file": c.file,
            "start_line": c.start_line, "end_line": c.end_line,
            "part": c.part, "n_parts": c.n_parts,
        } for c in chunks],
    )
    print(f"[embed] indexed into '{name}' at {CHROMA_DIR}")
    return col


def has_index(repo_name: str) -> bool:
    """Whether this repo has already been embedded, without loading the model."""
    try:
        col = _client().get_collection(collection_name(REPOS_DIR / repo_name))
        return col.count() > 0
    except Exception:
        return False


def search(repo_name: str, query: str, k: int = 5) -> list[dict]:
    """Top-k *distinct symbols*.

    A long function is several chunks, and they all score alike — without
    collapsing them one function would fill every slot. Over-fetch, then keep
    each symbol's best-matching chunk.
    """
    col = _client().get_collection(collection_name(REPOS_DIR / repo_name))
    vec = get_model().encode([query], normalize_embeddings=True)[0]
    res = col.query(query_embeddings=[vec.tolist()], n_results=min(k * 4, 100))

    best: dict[str, dict] = {}
    for i in range(len(res["ids"][0])):
        md = res["metadatas"][0][i]
        score = 1 - res["distances"][0][i]  # cosine distance -> similarity
        prev = best.get(md["qualname"])
        if prev is None or score > prev["score"]:
            best[md["qualname"]] = {
                "qualname": md["qualname"],
                "file": md["file"],
                "start_line": md["start_line"],
                "kind": md["kind"],
                "score": score,
                "part": f"{md['part']}/{md['n_parts']}",
            }
    return sorted(best.values(), key=lambda h: h["score"], reverse=True)[:k]


def _print_hits(query: str, hits: list[dict]) -> None:
    print(f"\n  query: {query!r}")
    for h in hits:
        part = "" if h["part"] == "1/1" else f"  [part {h['part']}]"
        print(f"    {h['score']:.3f}  {h['qualname']}")
        print(f"           {h['file']}:{h['start_line']}  ({h['kind']}){part}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 — embed a repo into ChromaDB")
    ap.add_argument("url", help="GitHub repo URL")
    ap.add_argument("-q", "--query", action="append", help="search instead of rebuilding")
    ap.add_argument("-k", type=int, default=5, help="results per query")
    args = ap.parse_args()

    repo_name = args.url.rstrip("/").removesuffix(".git").split("/")[-1]

    if args.query:
        for q in args.query:
            _print_hits(q, search(repo_name, q, args.k))
        return

    build_index(analyze(args.url))


if __name__ == "__main__":
    main()
