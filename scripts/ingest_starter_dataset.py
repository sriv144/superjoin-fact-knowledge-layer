"""
Script to ingest the curated starter dataset into SQLite database.
Extracts facts from key pages of all three Delhivery documents,
verifies grounding, normalizes metrics, matches cross-document candidates,
and computes relationships.
"""

import os
import sys
import time

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import FactKnowledgePipeline
from src.pdf_extractor import PDFExtractor
from src.matcher import CandidateMatcher
from src.models import RelationshipType

def main():
    print("==================================================")
    print("Ingesting Delhivery Starter Dataset...")
    print("==================================================")

    pdf_specs = {
        '01-delhivery-prospectus-2022-excerpt.pdf': [1, 47, 85, 88],
        '02-delhivery-annual-report-fy24-excerpt.pdf': [2, 36, 43, 51],
        '03-delhivery-q4-fy24-earnings-presentation.pdf': [6, 8]
    }

    db_path = "data/facts.db"
    pipeline = FactKnowledgePipeline(db_path=db_path)
    # Clear existing data for a fresh clean run
    pipeline.db.clear_all()

    total_facts = 0
    total_failures = 0

    for fname, pages in pdf_specs.items():
        fpath = os.path.join("starter-datasets/delhivery", fname)
        if not os.path.exists(fpath):
            print(f"Error: File not found: {fpath}")
            continue

        print(f"\n--- Ingesting {fname} (Pages {pages}) ---")
        meta, pages_content = PDFExtractor.extract_document(fpath, target_pages=pages)
        doc_facts = []
        doc_failures = []

        for p in pages_content:
            print(f"  Extracting Page {p.page_number} ({p.char_count} chars)...", end="", flush=True)
            t0 = time.time()
            facts, fails = pipeline.fact_extractor.extract_from_page(
                page_text=p.text,
                document_id=meta.id,
                source_filename=meta.filename,
                page_number=p.page_number
            )
            elapsed = time.time() - t0
            print(f" Done ({elapsed:.1f}s) -> {len(facts)} facts ({sum(1 for f in facts if f.is_grounded)} grounded), {len(fails)} failures")
            doc_facts.extend(facts)
            doc_failures.extend(fails)

        meta.extracted_facts_count = len(doc_facts)
        meta.grounded_facts_count = sum(1 for f in doc_facts if f.is_grounded)
        meta.status = "processed"

        pipeline.db.save_document(meta)
        pipeline.db.save_facts(doc_facts)
        for f in doc_failures:
            pipeline.db.save_failure_event(f)

        total_facts += len(doc_facts)
        total_failures += len(doc_failures)

    print("\n==================================================")
    print(f"Extraction Complete: {total_facts} total facts saved.")
    print("Computing Cross-Document Relationships...")
    print("==================================================")

    relationships = pipeline.compute_relationships()
    print(f"Identified {len(relationships)} cross-document relationships.")

    corroborates = [r for r in relationships if r.relationship_type == RelationshipType.CORROBORATES]
    contradicts = [r for r in relationships if r.relationship_type == RelationshipType.CONTRADICTS]
    reconcilable = [r for r in relationships if r.relationship_type == RelationshipType.RECONCILABLE]

    print(f"  - CORROBORATES: {len(corroborates)}")
    print(f"  - CONTRADICTS:  {len(contradicts)}")
    print(f"  - RECONCILABLE: {len(reconcilable)}")

    print("\nSample Relationships:")
    for r in relationships[:5]:
        print(f"\n[{r.relationship_type.value}] {r.fact_a.subject} | {r.fact_a.predicate} (Conf: {r.confidence})")
        print(f"  Doc A: {r.fact_a.source_filename} p.{r.fact_a.page_number} -> {r.fact_a.raw_value}")
        print(f"  Doc B: {r.fact_b.source_filename} p.{r.fact_b.page_number} -> {r.fact_b.raw_value}")
        print(f"  Reason: {r.short_explanation}")

if __name__ == "__main__":
    main()
