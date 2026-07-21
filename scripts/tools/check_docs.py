#!/usr/bin/env python3
"""Verify that doc files referencing source modules are not stale.

Each doc file can declare one or more code references via a comment:

    <!-- code-ref: engine/prefix_cache.py -->
    <!-- code-ref: engine/scheduler.py -->

This script checks that every referenced path still exists in the repo.
It does NOT check content consistency — only that the file exists.
This catches the most common drift case: a source file is deleted or
renamed but the doc is never updated.

Usage:
    python scripts/tools/check_docs.py          # check all docs/
    python scripts/tools/check_docs.py --fix    # print the stale refs (no auto-fix)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = ROOT / "docs"
REF_PATTERN = re.compile(r"<!--\s*code-ref:\s*(.+?)\s*-->")


def find_refs(doc: Path) -> list[tuple[Path, str]]:
    """Return (doc_path, ref) pairs for every code-ref in doc."""
    refs = []
    for line in doc.read_text(encoding="utf-8").splitlines():
        for m in REF_PATTERN.finditer(line):
            refs.append((doc, m.group(1).strip()))
    return refs


def check() -> int:
    docs = list(DOCS_DIR.rglob("*.md"))
    all_refs = []
    for doc in docs:
        all_refs.extend(find_refs(doc))

    if not all_refs:
        print("No code-ref annotations found in docs/.")
        return 0

    stale = [(doc, ref) for doc, ref in all_refs if not (ROOT / ref).exists()]

    if not stale:
        print(f"OK — {len(all_refs)} code-ref(s) in {len(docs)} docs, all valid.")
        return 0

    print(f"STALE code-refs ({len(stale)}/{len(all_refs)}):\n")
    for doc, ref in stale:
        print(f"  {doc.relative_to(ROOT)}  →  {ref}  (not found)")
    print(
        "\nUpdate the doc or the code-ref annotation. "
        "Source of truth is always the code."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="Same as default — prints stale refs, no auto-fix.")
    parser.parse_args()
    sys.exit(check())


if __name__ == "__main__":
    main()
