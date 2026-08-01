"""Regression tests for revision-markup parsing and minimum traceability."""

from pathlib import Path

from scripts.revision_markup_audit.mapping import requirement_for
from scripts.revision_markup_audit.spans import rone_spans, strip_rone


def test_strip_rone_handles_nested_latex_braces(tmp_path: Path) -> None:
    source = tmp_path / "sample.tex"
    source.write_text("\\section{Test}\n\\Rone{A $\\alpha_{m}$ value.}", encoding="utf-8")
    assert strip_rone(source.read_text(encoding="utf-8")) == "\\section{Test}\nA $\\alpha_{m}$ value."


def test_all_marked_generic_spans_have_a_requirement() -> None:
    source = Path("latex_restructure/main.tex")
    assert all(requirement_for(span).strip() for span in rone_spans(source))
