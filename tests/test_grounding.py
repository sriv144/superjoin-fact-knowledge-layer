"""
Tests for evidence grounding verification.
Verifies that claimed facts are backed by verbatim or near-verbatim text in source documents,
and that hallucinations or fabricated evidence quotes are strictly rejected.
"""

from src.grounding import verify_grounding, normalize_text_for_matching


def test_normalize_text_for_matching():
    raw = "Delhivery Limited’s   revenue \n was \r\n ₹8,142 Cr – FY24"
    normalized = normalize_text_for_matching(raw)
    assert "delhivery limited's revenue was rs8,142 cr - fy24" in normalized


def test_exact_grounding_success():
    page_text = (
        "Revenues from customers increased by 12.68% to ₹81,415.38 million for FY24 "
        "from ₹72,253.01 million for FY23. Express parcel shipment volumes increased by 11.48%."
    )
    quote = "Revenues from customers increased by 12.68% to ₹81,415.38 million for FY24"
    is_grounded, score, note = verify_grounding(page_text, quote)
    assert is_grounded is True
    assert score == 1.0


def test_line_break_and_whitespace_grounding_success():
    page_text = (
        "Delhivery In Numbers \n"
        "18,793\n(1)\nPin codes covered\n"
        "Workforce strength\n98,135\n(1,5)"
    )
    # Quote spanning line breaks with footnotes
    quote = "18,793 Pin codes covered"
    is_grounded, score, note = verify_grounding(page_text, quote)
    assert is_grounded is True
    assert score >= 0.8


def test_fabricated_quote_rejected():
    page_text = (
        "Delhivery Limited reported revenue from contracts with customers of ₹81,415.38 million. "
        "Pin codes covered was 18,793 across all states."
    )
    # Completely hallucinated quote not present in page
    fake_quote = "Delhivery operates a global network across 45,000 cities in North America"
    is_grounded, score, note = verify_grounding(page_text, fake_quote)
    assert is_grounded is False
    assert score < 0.5
    assert "Ungrounded" in note


def test_empty_inputs():
    is_grounded, score, note = verify_grounding("", "some quote")
    assert is_grounded is False
    assert score == 0.0

    is_grounded, score, note = verify_grounding("some text", "")
    assert is_grounded is False
    assert score == 0.0
