#!/usr/bin/env python3
"""
GVK Private AI — Генератор векторної бази пам'яті (tools/build_rag.py)

Компілює текстовий файл спогадів (`memories.txt`, `.md`, `.json`, `.jsonl`)
у векторний індекс `memory.faiss` (і супровідний `memories.json`) для
подальшого пакування в `payload.tar.enc` через `tools/disc_packer.py`.

Приклади запуску:
  # Пряма збірка (найзручніше):
  python3 tools/build_rag.py --input memories.txt --output raw/friend_1_data/memory.faiss

  # Збірка у папку + миттєва перевірка пошуком:
  python3 tools/build_rag.py -i memories.txt -o raw/friend_1_data/ --test-query "тестовий запит до пам'яті"

  # Через підкоманду build:
  python3 tools/build_rag.py build -i memories.txt -o raw/friend_1_data/memory.faiss

  # Автономна перевірка:
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


def _resolve_output_faiss_path(output_arg: str) -> str:
    """Якщо передано папку (існуючу або зі слешем у кінці), додає `memory.faiss`."""
    if output_arg.endswith(("/", "\\")) or os.path.isdir(output_arg):
        return os.path.join(output_arg, "memory.faiss")
    if not os.path.splitext(output_arg)[1]:
        return os.path.join(output_arg, "memory.faiss")
    return output_arg


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _run_build(args: argparse.Namespace) -> None:
    output_faiss = _resolve_output_faiss_path(args.output)
    print("====================================================================")
    print(" [*] GVK Private AI — RAG Builder (tools/build_rag.py)")
    print("====================================================================")
    print(f" Вхідний файл : {os.path.abspath(args.input)}")
    print(f" Вихідний індекс: {os.path.abspath(output_faiss)}")
    print(f" Модель       : {args.model} (backend={args.backend})")

    stats = build_rag_index(
        input_path=args.input,
        output_faiss_path=output_faiss,
        model_name=args.model,
        backend=args.backend,
    )

    print("--------------------------------------------------------------------")
    print(f" [✓] Спогадів оброблено : {stats['count']}")
    print(f" [✓] Активний бекенд    : {stats['backend']}")
    print(f" [✓] Формат індексу     : {stats['index_format']} (dim={stats['dimension']})")
    print(f" [✓] memory.faiss       : {stats['faiss_path']} ({_format_size(stats['faiss_bytes'])})")
    print(f" [✓] memories.json      : {stats['memories_json_path']} ({_format_size(stats['json_bytes'])})")

    if stats["backend"] == "ngram" and args.backend == "auto":
        print(
            "\n [i] Примітка: використано легкий офлайн-енкодер `ngram`, бо пакет "
            "`sentence-transformers` недоступний або модель не завантажена."
        )

    if args.test_query:
        print("\n--------------------------------------------------------------------")
        print(f" [?] Тестовий запит: {args.test_query!r}")
        rag = MemoryRAG(stats["faiss_path"])
        hits = rag.search(args.test_query, top_k=3)
        if not hits:
            print(" [!] Збігів вище порогу релевантності не знайдено.")
        else:
            for i, hit in enumerate(hits, 1):
                print(f"   #{i} (score={hit['score']:.4f} | {hit['category']}): {hit['text']}")

    print("====================================================================")


def _build_args_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="GVK Private AI — Збірка векторного індексу memory.faiss зі спогадів",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Шлях до вхідного файлу зі спогадами (.txt, .md, .json, .jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Шлях до вихідного файлу memory.faiss або папки персонажа",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Назва моделі ембедінгів (за замовчуванням: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "sentence-transformers", "ngram"),
        default="auto",
        help="Бекенд векторизації (auto — спроба sentence-transformers із фолбеком на ngram)",
    )
    parser.add_argument(
        "--test-query",
        default=None,
        help="Опціональний тестовий запит для перевірки пошуку одразу після збірки",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    if raw_argv and raw_argv[0] == "self-test":
        p_test = argparse.ArgumentParser(prog="build_rag.py self-test")
        p_test.add_argument(
            "--backend",
            choices=("auto", "sentence-transformers", "ngram"),
            default="ngram",
            help="Бекенд ембедінгів для self-test",
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
