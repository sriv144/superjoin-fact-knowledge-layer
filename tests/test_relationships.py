"""
Tests for relationship classification and contradiction safety.
Verifies that:
- Identical numbers with compatible context corroborate.
- Diverging numbers with differing time/scope are safely reconciled by context.
- Genuine discrepancies in identical context are identified as contradictions.
"""

from src.models import Fact, RelationshipType
from src.relationship_engine import RelationshipEngine


def test_corroboration_same_metric_and_period():
    fact_a = Fact(
        subject="Delhivery Limited",
        predicate="revenue_from_services",
        raw_value="₹81,415.38 million",
        normalized_value=8141.538,
        value_type="currency",
        unit="INR crore",
        time_period="FY2023-24",
        scope="consolidated",
        source_document_id="doc1",
        source_filename="02-delhivery-annual-report-fy24-excerpt.pdf",
        page_number=36,
        evidence_quote="Revenues from customers increased by 12.68% to ₹81,415.38 million for FY24"
    )
    fact_b = Fact(
        subject="Delhivery Limited",
        predicate="revenue_from_services",
        raw_value="₹8,142 Cr",
        normalized_value=8142.0,
        value_type="currency",
        unit="INR crore",
        time_period="FY24",
        scope="consolidated",
        source_document_id="doc2",
        source_filename="03-delhivery-q4-fy24-earnings-presentation.pdf",
        page_number=6,
        evidence_quote="FY24 revenue from services: ₹8,142 Cr"
    )

    engine = RelationshipEngine(llm_client=None)
    rel = engine.evaluate_pair(fact_a, fact_b)
    assert rel is not None
    assert rel.relationship_type == RelationshipType.CORROBORATES
    assert rel.confidence >= 0.9


def test_reconcilable_differing_reporting_periods():
    # 17,488 PIN codes in 2021 vs 18,793 PIN codes in 2024
    fact_a = Fact(
        subject="Delhivery Limited",
        predicate="pin_codes_covered",
        raw_value="17,488",
        normalized_value=17488.0,
        value_type="numeric",
        unit="pin_codes",
        time_period="December 31, 2021",
        scope="express parcel",
        source_document_id="doc1",
        source_filename="01-delhivery-prospectus-2022-excerpt.pdf",
        page_number=47,
        evidence_quote="serviced 17,488 PIN codes for the nine months period ended December 31, 2021"
    )
    fact_b = Fact(
        subject="Delhivery Limited",
        predicate="pin_codes_covered",
        raw_value="18,793",
        normalized_value=18793.0,
        value_type="numeric",
        unit="pin_codes",
        time_period="March 31, 2024",
        scope="express parcel",
        source_document_id="doc2",
        source_filename="02-delhivery-annual-report-fy24-excerpt.pdf",
        page_number=2,
        evidence_quote="18,793 Pin codes covered as of March 31, 2024"
    )

    engine = RelationshipEngine(llm_client=None)
    rel = engine.evaluate_pair(fact_a, fact_b)
    assert rel is not None
    assert rel.relationship_type == RelationshipType.RECONCILABLE
    assert rel.contextual_difference is not None


def test_reconcilable_differing_scope():
    # 98,135 (including partner agents) vs 63,713 (excluding partner agents)
    fact_a = Fact(
        subject="Delhivery Limited",
        predicate="workforce_strength",
        raw_value="98,135",
        normalized_value=98135.0,
        value_type="numeric",
        unit="count",
        time_period="March 31, 2024",
        scope="includes partner agents",
        source_document_id="doc1",
        source_filename="02-delhivery-annual-report-fy24-excerpt.pdf",
        page_number=2,
        evidence_quote="Workforce strength 98,135 Includes permanent employees, contractual workers and last mile deliver partner agents"
    )
    fact_b = Fact(
        subject="Delhivery Limited",
        predicate="team_size",
        raw_value="63,713",
        normalized_value=63713.0,
        value_type="numeric",
        unit="count",
        time_period="March 31, 2024",
        scope="excludes partner agents",
        source_document_id="doc2",
        source_filename="03-delhivery-q4-fy24-earnings-presentation.pdf",
        page_number=8,
        evidence_quote="Team size 63,713 excluding partner agents, daily wage manpower and security guards"
    )

    engine = RelationshipEngine(llm_client=None)
    rel = engine.evaluate_pair(fact_a, fact_b)
    assert rel is not None
    assert rel.relationship_type == RelationshipType.RECONCILABLE
    assert "partner" in rel.contextual_difference.lower()


def test_genuine_contradiction():
    # Corporate address PIN code conflict: 122002 vs 122001
    fact_a = Fact(
        subject="Delhivery Limited",
        predicate="corporate_office_address",
        raw_value="Plot 5, Sector 44, Gurugram 122002",
        normalized_value="Plot 5, Sector 44, Gurugram 122002",
        value_type="text",
        source_document_id="doc1",
        source_filename="01-delhivery-prospectus-2022-excerpt.pdf",
        page_number=1,
        evidence_quote="Corporate Office: Plot 5, Sector 44, Gurugram 122002, Haryana, India"
    )
    fact_b = Fact(
        subject="Delhivery Limited",
        predicate="corporate_office_address",
        raw_value="Plot No. 5, Sector 44, Gurugram 122001",
        normalized_value="Plot No. 5, Sector 44, Gurugram 122001",
        value_type="text",
        source_document_id="doc2",
        source_filename="02-delhivery-annual-report-fy24-excerpt.pdf",
        page_number=51,
        evidence_quote="Corporate address: Plot No. 5, Sector 44, Gurugram, Haryana 122001"
    )

    engine = RelationshipEngine(llm_client=None)
    rel = engine.evaluate_pair(fact_a, fact_b)
    assert rel is not None
    assert rel.relationship_type == RelationshipType.CONTRADICTS
    assert "122002" in rel.short_explanation
    assert "122001" in rel.short_explanation
    assert "Likely / unresolved contradiction" in rel.short_explanation


def test_strict_tolerance_boundary():
    """Verify that the strict 0.5% tolerance differentiates corroboration from minor rounding."""
    engine = RelationshipEngine(llm_client=None)

    # 1. Very close value (0.05% diff): Corroborates
    fa = Fact(subject="Entity", predicate="revenue", raw_value="1000", normalized_value=1000.0, value_type="numeric", time_period="2024", source_document_id="d1", source_filename="f1.pdf", page_number=1, evidence_quote="1000")
    fb = Fact(subject="Entity", predicate="revenue", raw_value="1002", normalized_value=1002.0, value_type="numeric", time_period="2024", source_document_id="d2", source_filename="f2.pdf", page_number=1, evidence_quote="1002")
    rel_close = engine.evaluate_pair(fa, fb)
    assert rel_close.relationship_type == RelationshipType.CORROBORATES

    # 2. Borderline value (1.5% diff): Reconcilable (minor variation), NOT Corroboration
    fc = Fact(subject="Entity", predicate="revenue", raw_value="1015", normalized_value=1015.0, value_type="numeric", time_period="2024", source_document_id="d2", source_filename="f2.pdf", page_number=1, evidence_quote="1015")
    rel_divergent = engine.evaluate_pair(fa, fc)
    assert rel_divergent.relationship_type == RelationshipType.RECONCILABLE

    # 3. Material difference (15% diff): Contradicts
    fd = Fact(subject="Entity", predicate="revenue", raw_value="1150", normalized_value=1150.0, value_type="numeric", time_period="2024", source_document_id="d2", source_filename="f2.pdf", page_number=1, evidence_quote="1150")
    rel_contradict = engine.evaluate_pair(fa, fd)
    assert rel_contradict.relationship_type == RelationshipType.CONTRADICTS

