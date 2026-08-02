"""Every path named in docs/results_index.md must exist.

The index is the reproducibility claim made checkable -- it tells a reviewer
which script produced which number. A stale path in it is worse than no index:
it sends someone looking for evidence that is not there.

Result ARTIFACTS are checked only when present, since a fresh clone has not run
anything yet. Scripts, figures and Lean files must always exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INDEX = _ROOT / "docs" / "results_index.md"
_ALWAYS = ("scripts/", "lean/", "tests/")


def _referenced_paths() -> list[str]:
    text = _INDEX.read_text()
    # Backticked paths, plus figure names in the Figure column.
    paths = set(re.findall(r"`([^`]*?/[^`]*?)`", text))
    cleaned: set[str] = set()
    for raw in paths:
        candidate = raw.split()[0].rstrip(".,;")
        if candidate.startswith(("http", "-")):
            continue
        cleaned.add(candidate)
    return sorted(cleaned)


@pytest.mark.parametrize("relative", _referenced_paths())
def test_indexed_path_exists(relative: str) -> None:
    # Brace expansions like n{16,20,24,32}.jsonl name a set; check the stem.
    if "{" in relative:
        relative = relative[: relative.index("{")]
    target = _ROOT / relative
    if relative.startswith(_ALWAYS):
        assert target.exists(), f"{relative} is referenced by the index but missing"
    elif not target.exists() and not target.parent.exists():
        pytest.skip(f"{relative} not generated in this checkout")


def test_every_figure_in_the_index_exists() -> None:
    figures = set(re.findall(r"`([a-z0-9_]+\.pdf)`", _INDEX.read_text()))
    assert figures, "the index names no figures; did the table change shape?"
    for figure in figures:
        assert (_ROOT / "experiments" / "figures" / figure).exists(), figure
