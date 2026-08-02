"""Check every DOI-bearing citation against the publisher's own record.

D11 says nothing is cited without verification, but "verified" had been
interpreted as "the identifier resolves to the right paper". That is not enough:
four entries carried a PARAPHRASE or a TRUNCATION in quotation marks, presented
as the paper's title. A reader who looks up the DOI finds a different string
than the one we printed, which reads as fabrication whether or not it was.

This script closes that gap mechanically. For each entry with a DOI it fetches
the Crossref record and compares the title we print against the registered one,
after normalising case, punctuation and whitespace. Mismatches are reported;
nothing is rewritten automatically, because the fix sometimes belongs in the
reading list's prose rather than the title field.

arXiv-only entries are listed but not checked here -- the arXiv API refuses
scripted access from this environment, so those are verified by hand and
recorded in the reading list.

Run from workspace/:
    python scripts/audit_citations.py --reading ../reading.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from eon.bibliography import parse_reading_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("audit_citations")

_CROSSREF = "https://api.crossref.org/works/"


def _normalise(title: str) -> str:
    """Case, punctuation and whitespace differ harmlessly between records."""
    lowered = title.lower().replace("–", "-").replace("—", "-")
    lowered = lowered.replace("’", "'").replace("--", "-")
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _crossref_title(doi: str, *, timeout: float) -> str | None:
    request = urllib.request.Request(
        _CROSSREF + doi,
        headers={"User-Agent": "eon-citation-audit/1.0 (mailto:rylan1012@gmail.com)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        logger.warning("%s: lookup failed (%s)", doi, error)
        return None
    titles = payload.get("message", {}).get("title") or []
    return titles[0] if titles else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reading", default="../reading.txt")
    parser.add_argument(
        "--proposal",
        default="../paper/proposal.md",
        help="cross-reference which keys the document actually cites",
    )
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--delay", type=float, default=0.4, help="politeness delay")
    args = parser.parse_args()

    entries, _ = parse_reading_list(Path(args.reading).read_text())
    with_doi = [entry for entry in entries if entry.doi]
    arxiv_only = [entry for entry in entries if not entry.doi and entry.arxiv]
    neither = [entry for entry in entries if not entry.doi and not entry.arxiv]

    logger.info(
        "%d entries: %d with DOI (checked here), %d arXiv-only, %d with neither",
        len(entries), len(with_doi), len(arxiv_only), len(neither),
    )

    cited: set[str] = set()
    proposal = Path(args.proposal)
    if proposal.exists():
        cited = set(re.findall(r"@(R\d+[ab]?)", proposal.read_text()))
        known = {entry.key for entry in entries}
        dangling = sorted(cited - known)
        if dangling:
            logger.error("cited but absent from the reading list: %s", ", ".join(dangling))
            sys.exit(1)
        logger.info("%d keys cited by the proposal: %s", len(cited), ", ".join(sorted(cited)))

    mismatches: list[tuple[str, str, str]] = []
    unchecked: list[str] = []
    for entry in with_doi:
        registered = _crossref_title(entry.doi, timeout=args.timeout)
        time.sleep(args.delay)
        if registered is None:
            unchecked.append(entry.key)
            continue
        if _normalise(entry.title) != _normalise(registered):
            mismatches.append((entry.key, entry.title, registered))
            logger.error("%s TITLE MISMATCH", entry.key)
            logger.error("   ours      : %s", entry.title)
            logger.error("   registered: %s", registered)
        else:
            logger.info("%s ok", entry.key)

    # The cited set is the one that matters: an unverified title on an entry
    # nobody cites is a tidiness problem, not a correctness one.
    cited_arxiv = sorted(e.key for e in arxiv_only if e.key in cited)
    if cited_arxiv:
        logger.info(
            "CITED and arXiv-only -- title verified by hand, not by this tool: %s",
            ", ".join(cited_arxiv),
        )
    if arxiv_only:
        logger.info(
            "all arXiv-only entries: %s", ", ".join(e.key for e in arxiv_only)
        )
    if neither:
        logger.info("no identifier: %s", ", ".join(e.key for e in neither))
    if unchecked:
        logger.warning("could not reach Crossref for: %s", ", ".join(unchecked))

    if mismatches:
        logger.error("%d title mismatch(es) -- fix the reading list", len(mismatches))
        sys.exit(1)
    logger.info("every reachable DOI title matches the registered record")


if __name__ == "__main__":
    main()
