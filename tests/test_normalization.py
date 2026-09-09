"""
Tests for value, unit, and currency normalization.
Verifies that disparate representations of financial and operational figures
normalize into directly comparable numeric quantities.
"""

from src.normalizer import parse_numeric_value, normalize_fact_values


def test_million_to_crore_normalization():
    # 1 Crore = 10 Million -> 81415.38 / 10 = 8141.538 Crore
    val, unit, val_type = parse_numeric_value("₹81,415.38 million")
    assert val == 8141.538
    assert unit == "INR crore"
    assert val_type == "currency"


def test_crore_normalization():
    val, unit, val_type = parse_numeric_value("₹8,142 Cr")
    assert val == 8142.0
    assert unit == "INR crore"
    assert val_type == "currency"


def test_million_vs_crore_comparability():
    # After normalization, ₹81,415.38 million and ₹8,142 Cr are within 0.01% of each other
    val_a, _, _ = parse_numeric_value("₹81,415.38 million")
    val_b, _, _ = parse_numeric_value("₹8,142 Cr")
    rel_diff = abs(val_a - val_b) / max(val_a, val_b)
    assert rel_diff < 0.001  # less than 0.1% rounding difference


def test_percentage_normalization():
    val, unit, val_type = parse_numeric_value("12.68%")
    assert val == 12.68
    assert unit == "percent"
    assert val_type == "percentage"


def test_comma_number_normalization():
    val, unit, val_type = parse_numeric_value("18,793")
    assert val == 18793.0
    assert val_type == "numeric"


def test_thousands_with_k_tonnes():
    val, unit, val_type = parse_numeric_value("1,429K tonnes")
    assert val == 1.429
    assert unit == "million tonnes"


def test_pdf_thousands_separator_artifact_in_currency_table():
    """A table text layer may turn visual 8,932 into 8.932; retain the table's scale."""
    val, unit, val_type = normalize_fact_values(
        "8.932",
        raw_unit="₹ Cr",
        evidence_quote="₹ Cr FY23 FY24 FY25 Revenue from services 7,224 8,142 8.932",
    )
    assert val == 8932.0
    assert unit == "INR crore"
    assert val_type == "currency"


def test_pdf_thousands_separator_uses_page_context():
    val, unit, _ = normalize_fact_values(
        "8.932",
        raw_unit="INR crore",
        evidence_quote="Revenue from services 8.932",
        source_context="₹ Cr FY23 FY24 FY25 Revenue from services 7,224 8,142 8.932",
    )
    assert val == 8932.0
    assert unit == "INR crore"


def test_malformed_rupee_glyph_with_million_unit():
    """PyMuPDF may expose a rupee glyph as J in otherwise valid Indian tables."""
    val, unit, val_type = parse_numeric_value("J89,319Mn")
    assert val == 8931.9
    assert unit == "INR crore"
    assert val_type == "currency"


def test_text_normalization_fallback():
    val, unit, val_type = normalize_fact_values("Plot No. 5, Sector 44, Gurugram 122001")
    assert val_type == "text"
    assert "Plot No. 5, Sector 44, Gurugram 122001" in str(val)
