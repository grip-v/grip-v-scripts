"""EXPANSION_PROMPTS.md Prompt 0, item 3: walk every sub-benchmark, grep the
`RESULT:` line from each `validation_report.txt`, and print a suite-wide
table (dataset, path, samples, questions, mismatches).

Walks the whole repo (excluding .git/.venv/common/tools) for any file named
`validation_report.txt` -- a dataset may have more than one (its main run
plus per-preset regression runs under tests/*_sample/), so this reports
every one found, not just one per dataset.

Usage:
    python tools/report_suite.py [repo_root]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RESULT_RE = re.compile(
    r"RESULT:\s*(\d+)\s*samples,\s*(\d+)\s*questions,\s*(\d+)\s*mismatches"
)
EXCLUDED_DIR_NAMES = {".git", ".venv", "__pycache__", "common", "tools", ".manim_build"}


def find_reports(repo_root: Path) -> list[Path]:
    found = []
    for path in repo_root.rglob("validation_report.txt"):
        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def parse_report(path: Path) -> dict | None:
    text = path.read_text(encoding="utf8", errors="replace")
    m = None
    for m in RESULT_RE.finditer(text):
        pass  # keep the LAST match -- a report may quote an earlier example
    if m is None:
        return None
    n_samples, n_questions, n_mismatches = (int(g) for g in m.groups())
    return dict(n_samples=n_samples, n_questions=n_questions, n_mismatches=n_mismatches)


def dataset_and_preset(repo_root: Path, path: Path) -> tuple[str, str]:
    rel = path.relative_to(repo_root)
    parts = rel.parts
    dataset = parts[0] if parts else "?"
    preset = parts[-2] if len(parts) >= 2 and parts[-2] != dataset else "(root)"
    return dataset, preset


def build_table(repo_root: Path) -> list[dict]:
    rows = []
    for path in find_reports(repo_root):
        parsed = parse_report(path)
        dataset, preset = dataset_and_preset(repo_root, path)
        rel = str(path.relative_to(repo_root))
        if parsed is None:
            rows.append(dict(dataset=dataset, preset=preset, path=rel,
                              n_samples=None, n_questions=None, n_mismatches=None,
                              status="NO RESULT: LINE"))
        else:
            rows.append(dict(dataset=dataset, preset=preset, path=rel, **parsed,
                              status="PASS" if parsed["n_mismatches"] == 0 else "FAIL"))
    return rows


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "No validation_report.txt files found."
    headers = ["dataset", "preset", "samples", "questions", "mismatches", "status", "path"]
    lines = ["\t".join(headers)]
    for r in rows:
        cells = [r["dataset"], r["preset"], r["n_samples"], r["n_questions"],
                  r["n_mismatches"], r["status"], r["path"]]
        lines.append("\t".join(str(c) for c in cells))
    total_samples = sum(r["n_samples"] for r in rows if r["n_samples"] is not None)
    total_mismatches = sum(r["n_mismatches"] for r in rows if r["n_mismatches"] is not None)
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    lines.append("")
    lines.append(f"{len(rows)} report(s), {n_pass} passing, "
                  f"{total_samples} total samples, {total_mismatches} total mismatches")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", nargs="?", default=str(ROOT))
    args = ap.parse_args()
    rows = build_table(Path(args.repo_root))
    print(format_table(rows))
    any_fail = any(r["status"] != "PASS" for r in rows)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
