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
from itertools import takewhile

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
# Longest run of gloss we will accept as a venue. Anything longer is prose.
_MAX_VENUE = 80
_QUARANTINE_HEADING = "QUARANTINED"
# Some R-keys are a CLAIM, not a paper: prose followed by "- Author (year),
# "Title," ... " anchor bullets, one per supporting paper. Flattening those
# marries the first anchor's eprint to the last anchor's title and DOI, which
# describes no real paper. Each anchor becomes its own child key (R39a, R39b).
_ANCHOR = re.compile(r"^\s*-\s+(\S.*)$")
# "R6 (above) ..." points back at an earlier definition; it is not an entry.
_CROSS_REFERENCE = re.compile(r"^\((above|see\b)")


@dataclass(frozen=True, slots=True)
class BibEntry:
    key: str
    authors: str
    year: str
    title: str
    arxiv: str | None
    doi: str | None
    note: str
    journal: str | None = None

    def to_bibtex(self) -> str:
        fields = [f"  author = {{{self.authors}}}", f"  title = {{{self.title}}}"]
        if self.year:
            fields.append(f"  year = {{{self.year}}}")
        if self.arxiv:
            fields.append(f"  eprint = {{{self.arxiv}}}")
            fields.append("  archivePrefix = {arXiv}")
        if self.journal:
            fields.append(f"  journal = {{{self.journal}}}")
        if self.doi:
            fields.append(f"  doi = {{{self.doi}}}")
        entry_type = "@article" if self.doi else "@misc"
        return f"{entry_type}{{{self.key},\n" + ",\n".join(fields) + "\n}\n"


class QuarantinedCitationError(RuntimeError):
    """Raised when an entry on the do-not-cite list would be emitted."""


class MalformedEntryError(RuntimeError):
    """Raised when an entry cannot be parsed into a real citation.

    Guessing here is the dangerous option: a wrong author or a title-shaped
    author field emits a reference to a paper that does not exist, which is the
    precise failure D11 exists to prevent. Fail the build instead and make the
    reading list say what it means.
    """


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
            entries.extend(_build_entries(current_key, current_lines))
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
    return sorted(best.values(), key=_sort_key)


def _sort_key(entry: BibEntry) -> tuple[int, str]:
    """Numeric by R-number, then by any anchor suffix (R39a before R39b)."""
    digits = "".join(takewhile(str.isdigit, entry.key[1:]))
    return int(digits), entry.key[1 + len(digits) :]


def _completeness(entry: BibEntry) -> tuple[int, int, int]:
    """A parsed author outranks a longer gloss: the duplicate is usually a
    back-reference whose commentary is longer than the definition's."""
    has_real_title = 0 if entry.title == entry.key else 1
    has_author = 0 if entry.authors == "Unknown" else 1
    return has_real_title, has_author, len(entry.note)


def normalize_authors(raw: str) -> str:
    """Join a surname list with BibTeX's " and " separator.

    The reading list writes authors as "A, B, C"; BibTeX reads a comma as a
    "Last, First" separator and mangles that into one malformed name.

    "et al." becomes the literal BibTeX name "others", which is how BibTeX and
    CSL spell an elided author list. Left as-is it parses as First="Lima et",
    Last="al." and renders as "al., Lima et."
    """
    cleaned = raw.strip().rstrip(",")
    if not cleaned:
        return cleaned
    cleaned = re.sub(r",?\s*\bet\.?\s+al\.?", ", others", cleaned).rstrip(",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) <= 1:
        return cleaned
    return " and ".join(parts)


def _is_continuation(line: str, entry_indent: int) -> bool:
    """A wrapped line of the current entry: indented deeper than the key."""
    stripped = line.lstrip()
    return bool(stripped) and (len(line) - len(stripped)) > entry_indent


def _split_anchors(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Partition an entry's lines into its own text and its anchor bullets."""
    head: list[str] = []
    anchors: list[list[str]] = []
    for line in lines:
        bullet = _ANCHOR.match(line)
        if bullet is not None:
            anchors.append([bullet.group(1)])
        elif anchors:
            anchors[-1].append(line)
        else:
            head.append(line)
    return head, anchors


def _build_entries(key: str, lines: list[str]) -> list[BibEntry]:
    """One reading-list entry yields one citation, or one per anchor bullet."""
    if lines and _CROSS_REFERENCE.match(lines[0]):
        return []  # "R6 (above) ..." -- a pointer at a definition elsewhere
    head, anchors = _split_anchors(lines)
    if anchors:
        # The parent is prose stating a claim; the anchors are the papers.
        return [
            entry
            for index, anchor in enumerate(anchors)
            if (entry := _build_entry(f"{key}{chr(ord('a') + index)}", " ".join(anchor)))
            is not None
        ]
    entry = _build_entry(key, " ".join(head))
    return [entry] if entry is not None else []


def _extract_journal(body: str, title_match: re.Match[str] | None) -> str | None:
    """The venue, when the entry has one, sits IMMEDIATELY after the title.

    Anchoring there rather than scanning forward is the whole trick. A preprint
    reads `"Title," arXiv:1234.5678. Commentary...`, and a forward scan for
    something capitalised skips the identifier and files the commentary as a
    journal -- printing our own notes as a publication venue. Nothing is
    extracted unless it starts where a venue must start.

    The end is the identifier or a semicolon, never a period: venues are full
    of abbreviating periods ("IEEE Trans. Power Systems") and cutting at the
    first one leaves a fragment.
    """
    if title_match is None:
        return None
    rest = body[title_match.end():].lstrip(" ,")
    if not rest or not rest[0].isupper():
        return None  # an identifier or lowercase gloss follows, not a venue
    for terminator in ("; ", " DOI ", " arXiv:", "; DOI"):
        index = rest.find(terminator)
        if index != -1:
            rest = rest[:index]
    # Only break at a sentence period once a digit has appeared: a venue runs
    # out at its volume/page numerals, and every period before those belongs to
    # an abbreviation ("IEEE Trans.", "Renew. Sustain. Energy Rev.").
    seen_digit = False
    for position, character in enumerate(rest):
        if character.isdigit():
            seen_digit = True
        elif (
            seen_digit
            and character == "."
            and rest[position + 1 : position + 2] == " "
        ):
            rest = rest[:position]
            break
    candidate = rest.strip().rstrip(",.")
    if not candidate or len(candidate) > _MAX_VENUE or " -- " in candidate:
        return None
    return candidate


def _build_entry(key: str, body: str) -> BibEntry | None:
    title_match = _TITLE.search(body)
    year_match = _YEAR.search(body)
    arxiv_match = _ARXIV.search(body)
    doi_match = _DOI.search(body)
    if title_match is None and arxiv_match is None and doi_match is None:
        return None

    # Everything before the "(year)" is the author list -- unless the entry
    # opens with its title, in which case there is no author list to take.
    lead = body.split("(")[0]
    title = title_match.group(1).strip().rstrip(",") if title_match else key
    if title_match is not None and title_match.start() < len(lead):
        lead = ""
    authors = normalize_authors(lead)
    journal = _extract_journal(body, title_match)
    return BibEntry(
        key=key,
        authors=authors or "Unknown",
        year=year_match.group(1) if year_match else "",
        title=title,
        arxiv=arxiv_match.group(1) if arxiv_match else None,
        doi=doi_match.group(1) if doi_match else None,
        note=body,
        journal=journal,
    )


def _validate(entry: BibEntry) -> None:
    if entry.authors == "Unknown" or not entry.authors:
        raise MalformedEntryError(
            f"{entry.key}: no author list parsed. The reading list entry must "
            'read "Authors (year), \\"Title,\\" identifier" -- see PLAN.txt D11.'
        )
    if entry.title == entry.key:
        raise MalformedEntryError(f"{entry.key}: no quoted title parsed")
    if entry.authors == entry.title or entry.title in entry.authors:
        raise MalformedEntryError(
            f"{entry.key}: the title was parsed as the author, which would emit "
            "a reference to a paper that does not exist"
        )


def render_bibliography(text: str) -> tuple[str, list[BibEntry], set[str]]:
    """Render BibTeX, refusing to emit anything on the quarantine list."""
    entries, quarantined = parse_reading_list(text)
    for entry in entries:
        _validate(entry)
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
