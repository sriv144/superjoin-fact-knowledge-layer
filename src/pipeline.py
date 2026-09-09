"""
Pipeline orchestration module for the Fact Knowledge Layer.
Coordinates PDF extraction, fact extraction, grounding, candidate matching,
relationship reasoning, and SQLite persistence.
Supports full ingestion and incremental document addition.
"""

from typing import List, Dict, Optional, Tuple, Any
import os
import re
from src.models import Fact, CrossDocumentRelationship, DocumentMetadata, ExtractionFailureEvent
from src.pdf_extractor import PDFExtractor, PageContent
from src.fact_extractor import FactExtractor
from src.matcher import CandidateMatcher
from src.relationship_engine import RelationshipEngine
from src.database import Database
from src.llm_client import LLMClient


def select_salient_pages(pages_content: List[PageContent], max_pages: int = 15) -> List[PageContent]:
    """
    Intelligently select pages with high density of quantitative, corporate,
    and operational facts for fast and cost-effective extraction.
    Always includes first few pages (covers, summaries) and tables.
    """
    if len(pages_content) <= max_pages:
        return pages_content

    scored_pages = []
    keywords = [
        "revenue", "turnover", "ebitda", "profit", "workforce", "employee", "team size",
        "pin code", "pincode", "shipment", "parcel", "tonnage", "freight", "active customer",
        "corporate office", "registered office", "director", "resigned", "gdp", "inflation"
    ]

    for p in pages_content:
        score = 0
        text_lower = p.text.lower()
        if p.char_count < 100:
            continue
        # Check presence of numbers
        num_count = len(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", p.text))
        score += min(num_count, 30)

        # Keyword matching
        for kw in keywords:
            if kw in text_lower:
                score += 5

        # Priority for first 3 pages and pages with footnotes
        if p.page_number <= 3:
            score += 25
        if "(1)" in p.text or "footnote" in text_lower or "as of march 31" in text_lower:
            score += 15

        scored_pages.append((score, p))

    scored_pages.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in scored_pages[:max_pages]]
    selected.sort(key=lambda p: p.page_number)
    return selected


class FactKnowledgePipeline:
    """End-to-end pipeline for the Fact Knowledge Layer."""

    def __init__(
        self,
        db_path: str = "data/facts.db",
        llm_client: Optional[LLMClient] = None
    ):
        self.db = Database(db_path)
        self.llm = llm_client or LLMClient()
        self.fact_extractor = FactExtractor(self.llm)
        self.relationship_engine = RelationshipEngine(self.llm)

    def process_pdf(
        self,
        file_path: str,
        target_pages: Optional[List[int]] = None,
        max_salient_pages: Optional[int] = 12
    ) -> Tuple[DocumentMetadata, List[Fact], List[ExtractionFailureEvent]]:
        """
        Process a single PDF: extract text, extract facts, ground them, and save to database.
        """
        meta, pages = PDFExtractor.extract_document(file_path, target_pages=target_pages)
        
        if target_pages:
            selected_pages = pages
        elif max_salient_pages and len(pages) > max_salient_pages:
            selected_pages = select_salient_pages(pages, max_pages=max_salient_pages)
        else:
            selected_pages = pages

        all_facts: List[Fact] = []
        all_failures: List[ExtractionFailureEvent] = []

        for p in selected_pages:
            facts, failures = self.fact_extractor.extract_from_page(
                page_text=p.text,
                document_id=meta.id,
                source_filename=meta.filename,
                page_number=p.page_number
            )
            all_facts.extend(facts)
            all_failures.extend(failures)

        meta.extracted_facts_count = len(all_facts)
        meta.grounded_facts_count = sum(1 for f in all_facts if f.is_grounded)
        meta.status = "processed"

        # Persist document metadata, facts, and failure events
        self.db.save_document(meta)
        self.db.save_facts(all_facts)
        for fail in all_failures:
            self.db.save_failure_event(fail)

        return meta, all_facts, all_failures

    def compute_relationships(self) -> List[CrossDocumentRelationship]:
        """
        Compute relationships across all documents currently in the database.
        Idempotent: clears previous relationship state before saving newly computed pairs.
        """
        # Relationship claims should only be made from source-grounded facts.
        all_facts = self.db.get_facts(grounded_only=True)
        candidate_pairs = CandidateMatcher.find_candidate_pairs(all_facts)
        
        relationships: List[CrossDocumentRelationship] = []
        for fact_a, fact_b in candidate_pairs:
            rel = self.relationship_engine.evaluate_pair(fact_a, fact_b)
            if rel:
                relationships.append(rel)

        # Clear previous relationships to ensure exact idempotency
        with self.db._get_connection() as conn:
            conn.cursor().execute("DELETE FROM relationships")
            conn.commit()

        self.db.save_relationships(relationships)
        return relationships

    def run_full_pipeline(
        self,
        pdf_paths: List[str],
        page_specs: Optional[Dict[str, List[int]]] = None,
        max_salient_pages: Optional[int] = 12,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline over a list of PDF file paths.
        """
        processed_docs = []
        total_facts = 0
        total_failures = 0

        for path in pdf_paths:
            fname = os.path.basename(path)
            target_pgs = page_specs.get(fname) if page_specs else None
            meta, facts, failures = self.process_pdf(
                path,
                target_pages=target_pgs,
                max_salient_pages=max_salient_pages,
            )
            processed_docs.append(meta)
            total_facts += len(facts)
            total_failures += len(failures)

        relationships = self.compute_relationships()

        return {
            "documents_processed": len(processed_docs),
            "facts_extracted": total_facts,
            "relationships_found": len(relationships),
            "failures_recorded": total_failures
        }
