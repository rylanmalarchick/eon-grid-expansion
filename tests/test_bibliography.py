"""D11 is enforced mechanically here, not by memory.

The reading list keeps a QUARANTINED section of identifiers that must never be
cited (they came from an LLM research dump with unreliable arXiv IDs). These
tests pin that a quarantined identifier cannot reach the .bib even if someone
later adds an R-key for it.
"""

from __future__ import annotations

import pytest

from eon.bibliography import (
    QuarantinedCitationError,
    parse_reading_list,
    render_bibliography,
)

_SAMPLE = """READING -- test fixture
=======================

1. PRIMARY LITERATURE  [V]
==========================

  R1  Cuenca et al. (2025), "Event-informed identification of planning
      candidates," IEEE Trans. Power Systems 40(1) 492-504; DOI
      10.1109/TPWRS.2024.3404115. Layer A candidate generation.
  R5  Koch et al. (2025), "QOBLIB -- the Intractable Decathlon,"
      arXiv:2504.03832. Hardness benchmarking protocol.

6. QUARANTINED -- DO NOT CITE  [U]
==================================

    - arXiv:2602.14327 (claimed Clifford-based QAOA init) -- NOT verified.
    - arXiv:2602.04676 (claimed TN pre-optimization) -- NOT verified.
"""


_INDENT_VARIANTS = """READING -- indent fixture
=========================

1. MIXED INDENTATION  [V]
=========================

  R1  Two spaces, two after key (2025), "Style A," arXiv:2504.03832. Gloss.
  R2 Two spaces, one after key (2024), "Style B," arXiv:2404.10693. Gloss.
    R40 Four spaces (2020), "Style C," arXiv:2007.09713. Gloss.
      R47 Six spaces (2019), "Style D," arXiv:1904.07358. Gloss.
"""


def test_parses_every_indentation_style_used_in_the_reading_list() -> None:
    """The real list uses four different indents; an entry silently dropped
    because of whitespace is a citation that never reaches the proposal."""
    entries, _ = parse_reading_list(_INDENT_VARIANTS)
    assert {entry.key for entry in entries} == {"R1", "R2", "R40", "R47"}
    by_key = {entry.key: entry for entry in entries}
    assert by_key["R40"].arxiv == "2007.09713"
    assert by_key["R47"].title == "Style D"


def test_parses_entries_and_quarantine() -> None:
    entries, quarantined = parse_reading_list(_SAMPLE)
    keys = {entry.key for entry in entries}
    assert keys == {"R1", "R5"}
    assert quarantined == {"2602.14327", "2602.04676"}


def test_fields_are_extracted() -> None:
    entries, _ = parse_reading_list(_SAMPLE)
    by_key = {entry.key: entry for entry in entries}
    assert by_key["R5"].arxiv == "2504.03832"
    assert by_key["R5"].year == "2025"
    assert "Intractable Decathlon" in by_key["R5"].title
    assert by_key["R1"].doi == "10.1109/TPWRS.2024.3404115"
    assert by_key["R1"].arxiv is None
    assert "Cuenca" in by_key["R1"].authors


def test_quarantined_ids_never_reach_the_bibliography() -> None:
    bibtex, _, quarantined = render_bibliography(_SAMPLE)
    assert quarantined
    for identifier in quarantined:
        assert identifier not in bibtex


def test_an_r_keyed_quarantined_entry_raises() -> None:
    """The failure mode this exists to prevent: someone promotes a quarantined
    paper to an R-key without re-verifying it."""
    poisoned = _SAMPLE.replace(
        '  R5  Koch et al. (2025), "QOBLIB -- the Intractable Decathlon,"\n'
        "      arXiv:2504.03832. Hardness benchmarking protocol.",
        '  R5  Someone (2026), "Clifford warm start," arXiv:2602.14327. Nope.',
    )
    with pytest.raises(QuarantinedCitationError, match="2602.14327"):
        render_bibliography(poisoned)


def test_bibtex_is_wellformed() -> None:
    bibtex, entries, _ = render_bibliography(_SAMPLE)
    assert bibtex.count("@") == len(entries)
    assert bibtex.count("{") == bibtex.count("}")
    assert "archivePrefix = {arXiv}" in bibtex


_CROSS_REFERENCED = """READING -- cross-reference fixture
==================================

1. PRIMARY  [V]
===============

  R5  Koch et al. (2025), "QOBLIB -- the Intractable Decathlon,"
      arXiv:2504.03832. Hardness benchmarking protocol.

9. LATER SECTION  [V]
=====================

  R30 Someone (2018), "Tree-width and contraction," arXiv:1807.04599. See also
      R5 Koch arXiv:2504.03832, which we use for the synthetic spine.
"""


def test_cross_references_do_not_create_duplicate_keys() -> None:
    """A later entry may mention an earlier key mid-sentence. Emitting it twice
    yields duplicate BibTeX keys, which breaks the document build."""
    bibtex, entries, _ = render_bibliography(_CROSS_REFERENCED)
    keys = [entry.key for entry in entries]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    assert bibtex.count("@") == len(set(keys))
    # The surviving R5 must be the real definition, not the cross-reference.
    by_key = {entry.key: entry for entry in entries}
    assert "Intractable Decathlon" in by_key["R5"].title


@pytest.mark.parametrize(
    "raw,expected",
    [
        # BibTeX reads a comma as "Last, First"; a comma-separated surname list
        # is therefore mangled unless it is joined with " and ".
        ("Baertschi, Eidenbenz", "Baertschi and Eidenbenz"),
        (
            "Ellinas, Chevalier, Chatzivasileiadis",
            "Ellinas and Chevalier and Chatzivasileiadis",
        ),
        ("Lazo and Watts", "Lazo and Watts"),
        ("Cuenca et al.", "Cuenca et al."),
        ("Koch et al.", "Koch et al."),
    ],
)
def test_author_lists_use_bibtex_and_separator(raw: str, expected: str) -> None:
    from eon.bibliography import normalize_authors

    assert normalize_authors(raw) == expected
