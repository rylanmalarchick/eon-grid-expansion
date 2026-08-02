"""Turn the hand-maintained reading list into BibTeX.

PLAN.txt D11: nothing may be cited without an independent verification step,
and the reading list carries a QUARANTINED section of entries that must never
be cited. Enforcing that by memory is exactly how a bad arXiv ID reaches a
reviewer, so this module enforces it mechanically: quarantined identifiers are
collected first, and emitting one raises.

The reading list lives outside the code repository, so callers pass its path in
-- nothing here hardcodes a path to the planning notes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The reading list uses four indent styles (2/4/6 spaces, one or two
# spaces after the key). An entry dropped over whitespace is a citation
# that silently never reaches the proposal.
_ENTRY_START = re.compile(r"^(\s{2,6})(R\d+)\s+(.*)$")
_SECTION = re.compile(r"^\d+\.\s+(.*)$")
_ARXIV = re.compile(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5})")
# A DOI may contain periods but never ends with sentence punctuation.
_DOI = re.compile(r"DOI\s+(10\.[^\s,;]*[^\s,;.])")
_YEAR = re.compile(r"\((\d{4})\)")
_TITLE = re.compile(r"[\"“]([^\"”]+)[\"”]")
_QUARANTINE_HEADING = "QUARANTINED"


@dataclass(frozen=True, slots=True)
class BibEntry:
    key: str
    authors: str
    year: str
    title: str
    arxiv: str | None
    doi: str | None
    note: str

    def to_bibtex(self) -> str:
        fields = [f"  author = {{{self.authors}}}", f"  title = {{{self.title}}}"]
        if self.year:
            fields.append(f"  year = {{{self.year}}}")
        if self.arxiv:
            fields.append(f"  eprint = {{{self.arxiv}}}")
            fields.append("  archivePrefix = {arXiv}")
        if self.doi:
            fields.append(f"  doi = {{{self.doi}}}")
        entry_type = "@article" if self.doi else "@misc"
        return f"{entry_type}{{{self.key},\n" + ",\n".join(fields) + "\n}\n"


class QuarantinedCitationError(RuntimeError):
    """Raised when an entry on the do-not-cite list would be emitted."""


def parse_reading_list(text: str) -> tuple[list[BibEntry], set[str]]:
    """Returns (citable entries, quarantined identifiers).

    An entry is citable when it has an R-key and does NOT sit under the
    quarantine heading. Identifiers under that heading are returned so callers
    can assert none of them escaped.
    """
    entries: list[BibEntry] = []
    quarantined: set[str] = set()
    in_quarantine = False
    current_key: str | None = None
    current_lines: list[str] = []
    current_indent = 0

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key is not None and not in_quarantine:
            entry = _build_entry(current_key, " ".join(current_lines))
            if entry is not None:
                entries.append(entry)
        current_key, current_lines = None, []

    for raw_line in text.splitlines():
        section = _SECTION.match(raw_line.strip()) if raw_line.strip() else None
        if section is not None and raw_line.startswith(tuple("0123456789")):
            flush()
            in_quarantine = _QUARANTINE_HEADING in section.group(1).upper()
            continue

        start = _ENTRY_START.match(raw_line)
        if start is not None:
            flush()
            current_indent = len(start.group(1))
            current_key, current_lines = start.group(2), [start.group(3)]
            continue

        if in_quarantine:
            quarantined.update(_ARXIV.findall(raw_line))
            quarantined.update(_DOI.findall(raw_line))
        elif current_key is not None and _is_continuation(raw_line, current_indent):
            current_lines.append(raw_line.strip())
        elif not raw_line.strip():
            flush()
    flush()
    return _deduplicate(entries), quarantined


def _deduplicate(entries: list[BibEntry]) -> list[BibEntry]:
    """Keep one entry per key. A later entry may cite an earlier key
    mid-sentence, which parses as a second entry start; emitting both yields
    duplicate BibTeX keys and breaks the document build. The definition is the
    richer record (it carries a quoted title), so prefer that."""
    best: dict[str, BibEntry] = {}
    for entry in entries:
        incumbent = best.get(entry.key)
        if incumbent is None or _completeness(entry) > _completeness(incumbent):
            best[entry.key] = entry
    return sorted(best.values(), key=lambda entry: int(entry.key[1:]))


def _completeness(entry: BibEntry) -> tuple[int, int]:
    has_real_title = 0 if entry.title == entry.key else 1
    return has_real_title, len(entry.note)


def _is_continuation(line: str, entry_indent: int) -> bool:
    """A wrapped line of the current entry: indented deeper than the key."""
    stripped = line.lstrip()
    return bool(stripped) and (len(line) - len(stripped)) > entry_indent


def _build_entry(key: str, body: str) -> BibEntry | None:
    title_match = _TITLE.search(body)
    year_match = _YEAR.search(body)
    arxiv_match = _ARXIV.search(body)
    doi_match = _DOI.search(body)
    if title_match is None and arxiv_match is None and doi_match is None:
        return None

    authors = body.split("(")[0].strip().rstrip(",")
    title = title_match.group(1).strip().rstrip(",") if title_match else key
    return BibEntry(
        key=key,
        authors=authors or "Unknown",
        year=year_match.group(1) if year_match else "",
        title=title,
        arxiv=arxiv_match.group(1) if arxiv_match else None,
        doi=doi_match.group(1) if doi_match else None,
        note=body,
    )


def render_bibliography(text: str) -> tuple[str, list[BibEntry], set[str]]:
    """Render BibTeX, refusing to emit anything on the quarantine list."""
    entries, quarantined = parse_reading_list(text)
    for entry in entries:
        identifier = entry.arxiv or entry.doi
        if identifier and identifier in quarantined:
            raise QuarantinedCitationError(
                f"{entry.key} carries quarantined identifier {identifier}; "
                "the reading list marks it do-not-cite (PLAN.txt D11)"
            )
    header = (
        "% Generated from the project reading list -- do not edit by hand.\n"
        "% Entries under the reading list's QUARANTINED heading are excluded\n"
        "% by construction (PLAN.txt D11).\n\n"
    )
    return header + "\n".join(entry.to_bibtex() for entry in entries), entries, quarantined
