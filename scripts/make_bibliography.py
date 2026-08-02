"""Generate references.bib from the project reading list.

Paths are arguments: the reading list is a planning document that lives
outside this repository, and repo files never point back at it.

Run from workspace/:
    python scripts/make_bibliography.py --reading ../reading.txt \
        --out ../paper/references.bib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eon.bibliography import render_bibliography

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_bibliography")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reading", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bibtex, entries, quarantined = render_bibliography(Path(args.reading).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bibtex)

    with_arxiv = sum(1 for entry in entries if entry.arxiv)
    with_doi = sum(1 for entry in entries if entry.doi)
    logger.info(
        "wrote %d entries to %s (%d arXiv, %d DOI)",
        len(entries), out_path, with_arxiv, with_doi,
    )
    logger.info("excluded %d quarantined identifiers: %s",
                len(quarantined), ", ".join(sorted(quarantined)) or "none")
    missing = [e.key for e in entries if not e.arxiv and not e.doi]
    if missing:
        logger.warning("entries with no arXiv id or DOI: %s", ", ".join(missing))


if __name__ == "__main__":
    main()
