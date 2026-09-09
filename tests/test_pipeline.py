"""
End-to-end smoke and integration tests for the Fact Knowledge Layer pipeline.
Verifies document persistence, cross-document candidate pairing, and failure logging.
"""

import os
import tempfile
from src.models import Fact, DocumentMetadata, ExtractionFailureEvent, RelationshipType
from src.database import Database
from src.matcher import CandidateMatcher
from src.relationship_engine import RelationshipEngine


def test_database_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_facts.db")
        db = Database(db_path)

        # 1. Document persistence
        doc = DocumentMetadata(
            id="doc_abc",
            filename="test_doc.pdf",
            page_count=10,
            processed_pages=2,
            extracted_facts_count=3,
            grounded_facts_count=3,
            status="processed"
        )
        db.save_document(doc)
        docs = db.get_documents()
        assert len(docs) == 1
        assert docs[0].filename == "test_doc.pdf"

        # 2. Fact persistence
        fact = Fact(
            subject="Test Corp",
            predicate="revenue",
            raw_value="₹1,000 Cr",
            normalized_value=1000.0,
            value_type="currency",
            unit="INR crore",
            time_period="FY24",
            source_document_id="doc_abc",
            source_filename="test_doc.pdf",
            page_number=1,
            evidence_quote="Revenue was ₹1,000 Cr in FY24"
        )
        db.save_facts([fact])
        facts = db.get_facts()
        assert len(facts) == 1
        assert facts[0].raw_value == "₹1,000 Cr"

        # 3. Failure event persistence
        fail = ExtractionFailureEvent(
            document_filename="test_doc.pdf",
            page_number=2,
            failure_type="table_misalignment",
            what_happened="Multi-column grid flattened into sequence",
            why_it_happened="PDF text extraction lacked spatial coordinates",
            how_handled="Flagged fact as ungrounded",
            future_improvement="Add bounding box table extractor"
        )
        db.save_failure_event(fail)
        events = db.get_failure_events()
        assert len(events) == 1
        assert events[0].failure_type == "table_misalignment"


def test_candidate_matcher_avoids_intra_document_matching():
    # Two facts from the SAME document should NEVER be paired
    fact_1 = Fact(
        subject="Company",
        predicate="revenue",
        raw_value="₹100 Cr",
        normalized_value=100.0,
        source_document_id="same_doc_123",
        source_filename="doc1.pdf",
        page_number=1,
        evidence_quote="rev 100"
    )
    fact_2 = Fact(
        subject="Company",
        predicate="revenue",
        raw_value="₹100 Cr",
        normalized_value=100.0,
        source_document_id="same_doc_123",
        source_filename="doc1.pdf",
        page_number=5,
        evidence_quote="rev 100"
    )
    pairs = CandidateMatcher.find_candidate_pairs([fact_1, fact_2])
    assert len(pairs) == 0


def test_candidate_matcher_pairs_cross_document():
    # Two facts from DIFFERENT documents should be paired
    fact_1 = Fact(
        subject="Company",
        predicate="revenue",
        raw_value="₹100 Cr",
        normalized_value=100.0,
        source_document_id="doc_a",
        source_filename="doc_a.pdf",
        page_number=1,
        evidence_quote="rev 100"
    )
    fact_2 = Fact(
        subject="Company",
        predicate="revenue",
        raw_value="₹100 Cr",
        normalized_value=100.0,
        source_document_id="doc_b",
        source_filename="doc_b.pdf",
        page_number=2,
        evidence_quote="rev 100"
    )
    pairs = CandidateMatcher.find_candidate_pairs([fact_1, fact_2])
    assert len(pairs) == 1


def test_candidate_matcher_prefers_closest_duplicate_metric_pair():
    fact_a_wrong = Fact(
        subject="Company", predicate="revenue", raw_value="100", normalized_value=100.0,
        source_document_id="doc_a", source_filename="doc_a.pdf", page_number=1, evidence_quote="100"
    )
    fact_a_match = Fact(
        subject="Company", predicate="revenue", raw_value="900", normalized_value=900.0,
        source_document_id="doc_a", source_filename="doc_a.pdf", page_number=1, evidence_quote="900"
    )
    fact_b = Fact(
        subject="Company", predicate="revenue", raw_value="901", normalized_value=901.0,
        source_document_id="doc_b", source_filename="doc_b.pdf", page_number=1, evidence_quote="901"
    )

    pairs = CandidateMatcher.find_candidate_pairs([fact_a_wrong, fact_a_match, fact_b])
    assert len(pairs) == 1
    assert fact_a_match in pairs[0]
    assert fact_b in pairs[0]


def test_llm_client_reports_unavailable_without_key():
    from src.llm_client import LLMClient
    import pytest

    client = LLMClient(api_key="")
    # When no key is configured in env or passed
    assert client.provider == "none"
    assert not client.is_available()
    with pytest.raises(RuntimeError) as exc_info:
        client.generate_json("test prompt")
    assert "NVIDIA_API_KEY" in str(exc_info.value)


def test_llm_client_nvidia_configuration():
    from src.llm_client import LLMClient

    # Test initialization with explicit mock key
    client = LLMClient(api_key="mock-test-key-for-config-validation")
    assert client.provider == "nvidia_nim"
    assert client.is_available()
    assert "nemotron" in client.model_name.lower() or "nvidia" in client.model_name.lower()
