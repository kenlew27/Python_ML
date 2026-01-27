#!/usr/bin/env python3
"""
score_headlines.py

Usage:
  python score_headlines.py <input_headlines_txt> <source>

Input:
  - UTF-8 text file with ONE headline per line.

Output:
  - headline_scores_<source>_<YYYY>_<MM>_<DD>.txt
    Each line: <PredictedLabel>,<OriginalHeadline>
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

import joblib


def _ensure_numpy_pickle_compat() -> None:
    """
    Some joblib pickles created with NumPy 2.x reference "numpy._core.*".
    If running on NumPy 1.x, those module paths don't exist. We alias them.
    """
    try:
        import numpy.core as np_core  # type: ignore

        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = np_core
            sys.modules["numpy._core._multiarray_umath"] = importlib.import_module(
                "numpy.core._multiarray_umath"
            )
            sys.modules["numpy._core.multiarray"] = importlib.import_module("numpy.core.multiarray")
            sys.modules["numpy._core.umath"] = importlib.import_module("numpy.core.umath")
    except Exception:
        # If this fails, joblib.load will raise a clearer error later.
        pass


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="score_headlines.py",
        description="Score headlines as Optimistic/Pessimistic/Neutral using a pre-trained SVM.",
    )
    p.add_argument("input_file", help="UTF-8 text file with one headline per line.")
    p.add_argument("source", help="Source label (e.g., nyt, chicagotribune, la_times).")
    return p.parse_args(argv)


def _read_headlines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln for ln in (s.strip() for s in lines) if ln]


def _find_model_path() -> Path:
    candidates = [Path("svm.joblib"), Path(__file__).resolve().parent / "svm.joblib"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 'svm.joblib' in the current directory or next to score_headlines.py."
    )


def _load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer("all-MiniLM-L6-v2")


def _output_path(source: str, day: dt.date | None = None) -> Path:
    day = day or dt.date.today()
    safe_source = source.strip()
    return Path(f"headline_scores_{safe_source}_{day:%Y_%m_%d}.txt")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    in_path = Path(args.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    headlines = _read_headlines(in_path)
    if not headlines:
        raise ValueError(f"No headlines found in: {in_path}")

    _ensure_numpy_pickle_compat()
    clf = joblib.load(_find_model_path())

    embedder = _load_embedder()
    embeddings = embedder.encode(headlines)
    preds = clf.predict(embeddings)

    out_path = _output_path(args.source)
    with out_path.open("w", encoding="utf-8") as f:
        for label, headline in zip(preds, headlines):
            f.write(f"{label},{headline}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
