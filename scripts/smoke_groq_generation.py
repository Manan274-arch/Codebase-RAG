"""Run one live citation-aware generation request through Groq."""

import sys

from src.generation.context import ContextBundle, EvidenceItem
from src.generation.generator import CitationValidationError, generate_answer
from src.generation.groq import GenerationBackendError, GroqTextGenerator

QUESTION = "How does calculate_total compute its return value, including tax?"


def _context() -> ContextBundle:
    total = EvidenceItem(
        citation_id="C1",
        evidence_id="src/pricing.py::0",
        source="src/pricing.py",
        chunk_index=0,
        page_content=(
            "def calculate_total(subtotal: float) -> float:\n"
            "    return subtotal + calculate_tax(subtotal)"
        ),
        start_line=1,
        end_line=2,
        origin="retrieved",
    )
    tax = EvidenceItem(
        citation_id="C2",
        evidence_id="src/tax.py::0",
        source="src/tax.py",
        chunk_index=0,
        page_content=(
            "def calculate_tax(subtotal: float) -> float:\n"
            "    return subtotal * 0.10"
        ),
        start_line=1,
        end_line=2,
        origin="relationship",
    )
    rendered_context = (
        "[C1] src/pricing.py:1-2\n"
        f"{total.page_content}\n\n"
        "[C2] src/tax.py:1-2\n"
        f"{tax.page_content}"
    )
    return ContextBundle((total, tax), rendered_context)


def main() -> int:
    """Run the live request and return a process status without exposing secrets."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = generate_answer(QUESTION, _context(), GroqTextGenerator())
    except (GenerationBackendError, CitationValidationError) as error:
        print("Generated answer: <unavailable>")
        print("Validated citation IDs: <none>")
        print(f"Smoke test succeeded: no ({error})")
        return 1

    print("Generated answer:")
    print(result.answer)
    citations = ", ".join(result.citation_ids) or "<none>"
    print(f"Validated citation IDs: {citations}")
    if not result.citation_ids:
        print("Smoke test succeeded: no (the answer contained no valid citation)")
        return 1
    print("Smoke test succeeded: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
