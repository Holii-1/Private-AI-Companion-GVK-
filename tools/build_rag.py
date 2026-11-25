#!/usr/bin/env python3
"""
GVK Private AI - RAG Memory Index Builder (tools/build_rag.py)

Compiles a structured memory source file (`.txt`, `.md`, `.json`, or `.jsonl`) into
a FAISS vector index (`memory.faiss`) and accompanying `memories.json` metadata for
packaging into `payload.tar.enc` via `tools/disc_packer.py`.

Key Responsibilities:
  - Resolve target output paths whether given a directory or an explicit `.faiss` file.
  - Delegate embedding and index generation to `core.memory_rag.build_rag_index`.
  - Optionally run a post-build smoke query against the freshly compiled index.
  - Expose an offline `self-test` subcommand for verifying the RAG pipeline.

Usage:
  python3 tools/build_rag.py --input memories.txt --output raw/friend_1_data/memory.faiss
  python3 tools/build_rag.py -i memories.txt -o raw/friend_1_data/ --test-query "test memory query"
  python3 tools/build_rag.py build -i memories.txt -o raw/friend_1_data/memory.faiss
  python3 tools/build_rag.py self-test
"""

import argparse
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.memory_rag import (
    DEFAULT_EMBEDDING_MODEL,
    MemoryRAG,
    build_rag_index,
    run_self_test,
)


# =====================================================================
# PATH & FORMATTING HELPERS
# =====================================================================
def _resolve_output_faiss_path(output_arg: str) -> str:
    """Resolve the destination `memory.faiss` path from a file or directory argument.

    If `output_arg` points to an existing directory, ends with a path separator, or has
    no file extension, `memory.faiss` is appended automatically.

    Args:
        output_arg: CLI `--output` path value.

    Returns:
        Resolved path ending in a target `.faiss` filename.
    """
    if output_arg.endswith(("/", "\\")) or os.path.isdir(output_arg):
        return os.path.join(output_arg, "memory.faiss")
    if not os.path.splitext(output_arg)[1]:
        return os.path.join(output_arg, "memory.faiss")
    return output_arg


def _format_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string (B, KB, or MB).

    Args:
        num_bytes: Size in bytes.

    Returns:
        Formatted size string.
    """
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


# =====================================================================
# BUILD EXECUTION & CLI
# =====================================================================
def _run_build(args: argparse.Namespace) -> None:
    """Compile the RAG index from parsed CLI arguments and optionally run a test query.

    Args:
        args: Parsed namespace from `_build_args_parser`.
    """
    output_faiss = _resolve_output_faiss_path(args.output)
    print("=====================================================================")
    print(" [*] GVK Private AI - RAG Builder (tools/build_rag.py)")
    print("=====================================================================")
    print(f" Input file    : {os.path.abspath(args.input)}")
    print(f" Output index  : {os.path.abspath(output_faiss)}")
    print(f" Embedding     : {args.model} (backend={args.backend})")

    stats = build_rag_index(
        input_path=args.input,
        output_faiss_path=output_faiss,
        model_name=args.model,
        backend=args.backend,
    )

    print("---------------------------------------------------------------------")
    print(f" [OK] Memories indexed : {stats['count']}")
    print(f" [OK] Active backend   : {stats['backend']}")
    print(f" [OK] Index format     : {stats['index_format']} (dim={stats['dimension']})")
    print(f" [OK] memory.faiss     : {stats['faiss_path']} ({_format_size(stats['faiss_bytes'])})")
    print(f" [OK] memories.json    : {stats['memories_json_path']} ({_format_size(stats['json_bytes'])})")

    if stats["backend"] == "ngram" and args.backend == "auto":
        print(
            "\n [*] Note: Fell back to the lightweight offline `ngram` encoder because "
            "`sentence-transformers` is unavailable or the model is not cached locally."
        )

    if args.test_query:
        print("\n---------------------------------------------------------------------")
        print(f" [*] Test query: {args.test_query!r}")
        rag = MemoryRAG(stats["faiss_path"])
        hits = rag.search(args.test_query, top_k=3)
        if not hits:
            print(" [!] No matches found above the relevance threshold.")
        else:
            for i, hit in enumerate(hits, 1):
                print(f"   #{i} (score={hit['score']:.4f} | {hit['category']}): {hit['text']}")

    print("=====================================================================")


def _build_args_parser(prog: str) -> argparse.ArgumentParser:
    """Create the argument parser for building a RAG memory index.

    Args:
        prog: Program name displayed in CLI usage output.

    Returns:
        Configured `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="GVK Private AI - Build a memory.faiss vector index from persona memories",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the source memories file (.txt, .md, .json, .jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination path for memory.faiss or a persona data directory",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model name (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "sentence-transformers", "ngram"),
        default="auto",
        help="Vectorization backend ('auto' tries sentence-transformers with an ngram fallback)",
    )
    parser.add_argument(
        "--test-query",
        default=None,
        help="Optional query to run against the built index for immediate verification",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the RAG index builder CLI.

    Supports direct invocation (`build_rag.py -i ... -o ...`), the explicit `build`
    subcommand, and the offline `self-test` subcommand.

    Args:
        argv: Optional list of CLI arguments (defaults to `sys.argv[1:]`).
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if raw_argv and raw_argv[0] == "self-test":
        p_test = argparse.ArgumentParser(
            prog="build_rag.py self-test",
            description="GVK Private AI - Offline self-test for RAG index building",
        )
        p_test.add_argument(
            "--backend",
            choices=("auto", "sentence-transformers", "ngram"),
            default="ngram",
            help="Embedding backend to use for self-test (default: ngram)",
        )
        test_args = p_test.parse_args(raw_argv[1:])
        run_self_test(backend=test_args.backend)
        return

    if raw_argv and raw_argv[0] == "build":
        parser = _build_args_parser(prog="build_rag.py build")
        args = parser.parse_args(raw_argv[1:])
        _run_build(args)
        return

    parser = _build_args_parser(prog="build_rag.py")
    args = parser.parse_args(raw_argv)
    _run_build(args)


if __name__ == "__main__":
    main()
