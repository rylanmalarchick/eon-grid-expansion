"""D11 is enforced mechanically here, not by memory.

The reading list keeps a QUARANTINED section of identifiers that must never be
cited (they came from an LLM research dump with unreliable arXiv IDs). These
tests pin that a quarantined identifier cannot reach the .bib even if someone
later adds an R-key for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eon.bibliography import (
    MalformedEntryError,
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


_CROSS_REF_WITH_ABOVE = """READING -- (above) fixture
==========================

1. PRIMARY  [V]
===============

  R6  Ellinas, Chevalier, Chatzivasileiadis (2024), "A hybrid Quantum-Classical
      Algorithm for Mixed-Integer Optimization in Power Systems,"
      arXiv:2404.10693. Hybrid Benders for power-system MILPs.

9. LATER SECTION  [V]
=====================

    R6 (above) Ellinas, Chevalier, Chatzivasileiadis (2024), arXiv:2404.10693.
        Accelerated Benders splits the MILP into an integer master + LP
        subproblem; the dual-cut machinery is the certificate's source. [V]
"""


def test_an_above_cross_reference_never_supplants_the_definition() -> None:
    """"R6 (above) ..." is a pointer, not an entry. Its body begins with a
    parenthetical, so naive author extraction yields the empty string -- and
    because the pointer's gloss is longer than the definition's, a
    length-based tiebreak silently promotes it and emits author={Unknown}."""
    _, entries, _ = render_bibliography(_CROSS_REF_WITH_ABOVE)
    by_key = {entry.key: entry for entry in entries}
    assert by_key["R6"].authors == "Ellinas and Chevalier and Chatzivasileiadis"


_TITLE_FIRST = """READING -- title-first fixture
==============================

5. HARDNESS  [V]
================

    R34 "Increasing the Hardness of Posiform Planting Using Random QUBOs,"
        arXiv:2411.03626 / npj Unconventional Computing (2025). Fuses random
        spin-glass Ising models into posiform-planted QUBOs. [V]
"""


def test_a_title_first_entry_refuses_to_invent_an_author() -> None:
    """When an entry opens with its title, everything before the first "(" is
    the title -- not an author list. Emitting it as the author produces a
    citation whose author field is the title, which is a fabricated reference."""
    with pytest.raises(MalformedEntryError, match="R34"):
        render_bibliography(_TITLE_FIRST)


_MULTI_ANCHOR = """READING -- multi-anchor fixture
===============================

8. CLAIMS  [V]
==============

    R39 Distribution-system RECONFIGURATION is strongly NP-hard; joint
        expansion+reconfiguration is the richest distribution sub-problem.
        Canonical anchors (verified from raw .tex / DOI 2026-06-27):
          - Khodabakhsh, Yang, Basu, Nikolova, Caramanis, Lianeas, Pountourakis
            (2017), "A Submodular Approach for Electricity Distribution Network
            Reconfiguration," arXiv:1711.03517 -- reconfiguration is STRONGLY
            NP-hard via a reduction from 3-PARTITION. [V]
          - Lavorato, Franco, Rider, Romero (2012), "Imposing Radiality
            Constraints in Distribution System Optimization Problems," IEEE
            TPWRS 27(1):172-180, DOI 10.1109/TPWRS.2011.2161349 -- the canonical
            fictitious-flow radiality encoding. [V]
"""


def test_a_multi_anchor_entry_yields_one_citation_per_anchor() -> None:
    """R39 is a CLAIM holding two distinct verified papers. Flattening it takes
    the first arXiv id, the last DOI, and the last title -- Khodabakhsh's eprint
    married to Lavorato's title and DOI. That reference describes no real paper."""
    _, entries, _ = render_bibliography(_MULTI_ANCHOR)
    by_key = {entry.key: entry for entry in entries}
    assert "R39" not in by_key, "the prose parent must not be emitted as a paper"
    assert set(by_key) == {"R39a", "R39b"}

    khodabakhsh = by_key["R39a"]
    assert khodabakhsh.arxiv == "1711.03517"
    assert khodabakhsh.doi is None
    assert khodabakhsh.authors.startswith("Khodabakhsh")

    lavorato = by_key["R39b"]
    assert lavorato.doi == "10.1109/TPWRS.2011.2161349"
    assert lavorato.arxiv is None
    assert "Radiality" in lavorato.title


def test_the_real_reading_list_emits_no_malformed_entry() -> None:
    """The regression guard: run the generator over the actual reading list.
    Every bug above reached references.bib before anyone looked at it."""
    reading = Path(__file__).resolve().parents[2] / "reading.txt"
    if not reading.exists():  # the code repo may be checked out standalone
        pytest.skip(f"reading list not present at {reading}")
    _, entries, _ = render_bibliography(reading.read_text())
    for entry in entries:
        assert entry.title != entry.key, f"{entry.key}: no title parsed"
        assert entry.authors != "Unknown", f"{entry.key}: no author parsed"


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
        # BibTeX spells "and the rest" as the literal name "others"; the
        # shorthand "Lima et al." is read as First="Lima et" Last="al.", which
        # renders in the bibliography as "al., Lima et."
        ("Cuenca et al.", "Cuenca and others"),
        ("Koch et al.", "Koch and others"),
        ("Pelofske, Baertschi, et al.", "Pelofske and Baertschi and others"),
    ],
)
def test_author_lists_use_bibtex_and_separator(raw: str, expected: str) -> None:
    from eon.bibliography import normalize_authors

    assert normalize_authors(raw) == expected


_WITH_VENUE = """READING -- venue fixture
========================

1. PRIMARY  [V]
===============

  R2  Lazo and Watts (2024), "Stochastic expansion via DistFlow," Renew.
      Sustain. Energy Rev. 191, 114156; DOI 10.1016/j.rser.2023.114156. Gloss.
  R50 Gust et al. (2024), "Designing electricity distribution networks,"
      European Journal of Operational Research 315(1):271-288. Gloss here.
  R47 Baertschi, Eidenbenz (2019), "Deterministic Preparation of Dicke
      States," arXiv:1904.07358 (LANL). Gloss.
"""


def test_journal_is_extracted_when_present() -> None:
    """An entry with no venue renders as a bare author-title-year line, which
    reads as a half-finished reference in a submitted document."""
    _, entries, _ = render_bibliography(_WITH_VENUE)
    by_key = {entry.key: entry for entry in entries}
    assert by_key["R50"].journal == "European Journal of Operational Research 315(1):271-288"
    assert "Renew" in (by_key["R2"].journal or "")
    # An arXiv preprint has no journal; inventing one would be a fabrication.
    assert by_key["R47"].journal is None


def test_journal_reaches_the_bibtex() -> None:
    bibtex, _, _ = render_bibliography(_WITH_VENUE)
    assert "journal = {European Journal of Operational Research 315(1):271-288}" in bibtex


_VENUE_TRAPS = """READING -- venue trap fixture
=============================

1. PRIMARY  [V]
===============

  R1  Cuenca et al. (2025), "Event-informed identification of planning
      candidates," IEEE Trans. Power Systems 40(1) 492-504; DOI
      10.1109/TPWRS.2024.3404115. Layer A candidate generation.
  R33 Hahn, Pelofske, Djidjev (2023), "Posiform Planting: Generating QUBO
      Instances for Benchmarking," arXiv:2308.05859. QUBOs of arbitrary size
      with a UNIQUE PLANTED optimum, tailored to a target connectivity.
      Resolves the NISQ-vs-hardness tension: hardware-native + known optimum.
  R2  Lazo and Watts (2024), "Stochastic expansion," Renew. Sustain. Energy
      Rev. 191, 114156; DOI 10.1016/j.rser.2023.114156. Gloss.
"""


def test_abbreviated_venue_is_not_truncated_at_its_periods() -> None:
    """"IEEE Trans. Power Systems" must survive; cutting at the first period
    after an abbreviation leaves the bare word "IEEE Trans" as the venue."""
    _, entries, _ = render_bibliography(_VENUE_TRAPS)
    by_key = {entry.key: entry for entry in entries}
    assert by_key["R1"].journal == "IEEE Trans. Power Systems 40(1) 492-504"
    assert by_key["R2"].journal == "Renew. Sustain. Energy Rev. 191, 114156"


def test_gloss_prose_never_becomes_a_venue() -> None:
    """A preprint entry is title, identifier, then commentary. Scanning forward
    for something capitalised finds the commentary and files it as a journal --
    printing a sentence of our own notes as if it were a publication venue."""
    _, entries, _ = render_bibliography(_VENUE_TRAPS)
    by_key = {entry.key: entry for entry in entries}
    assert by_key["R33"].journal is None, (
        f"invented a venue for a preprint: {by_key['R33'].journal!r}"
    )


def test_no_venue_in_the_real_bibliography_looks_like_prose() -> None:
    reading = Path(__file__).resolve().parents[2] / "reading.txt"
    if not reading.exists():
        pytest.skip("reading list not present")
    _, entries, _ = render_bibliography(reading.read_text())
    for entry in entries:
        if entry.journal is None:
            continue
        assert len(entry.journal) <= 80, f"{entry.key}: venue too long: {entry.journal!r}"
        assert " -- " not in entry.journal, f"{entry.key}: gloss leaked: {entry.journal!r}"
        assert not entry.journal.endswith("."), f"{entry.key}: {entry.journal!r}"
