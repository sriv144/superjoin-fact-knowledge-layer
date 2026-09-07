"""
Optional demo script to ingest the Delhivery starter dataset into SQLite.
By default, processes PDFs using generic pipeline extraction (up to max_salient_pages).
Pass --fast-demo to process only the pre-selected key demo pages for fast demonstration.

NOTE: Arbitrary user uploads in the Streamlit app use pipeline.process_pdf() directly
without any filename or page rules.
"""

import os
import sys
import time
import argparse

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import FactKnowledgePipeline
from src.pdf_extractor import PDFExtractor
from src.matcher import CandidateMatcher
from src.models import RelationshipType, Fact


def main():
    parser = argparse.ArgumentParser(description="Ingest starter dataset into SQLite database.")
    parser.add_argument(
        "--fast-demo",
        action="store_true",
        help="Process only pre-selected key demo pages for fast evaluation (default: generic salient pages)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="starter-datasets/delhivery",
        help="Path to folder containing PDF documents."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/facts.db",
        help="Target SQLite database file path."
    )
    args = parser.parse_args()

    print("==================================================")
    print("Superjoin Fact Knowledge Layer - Starter Ingestion")
    print(f"Mode: {'Fast Demo (Selected Pages)' if args.fast_demo else 'Generic Ingestion (All Documents)'}")
    print(f"Directory: {args.data_dir}")
    print(f"Database:  {args.db_path}")
    print("==================================================")

    pipeline = FactKnowledgePipeline(db_path=args.db_path)
    pipeline.db.clear_all()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    pdf_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".pdf")])
    if not pdf_files:
        print(f"No PDF files found in {args.data_dir}")
        sys.exit(1)

    demo_specs = {
        '01-delhivery-prospectus-2022-excerpt.pdf': [1, 47, 85, 88],
        '02-delhivery-annual-report-fy24-excerpt.pdf': [2, 36, 43, 51],
        '03-delhivery-q4-fy24-earnings-presentation.pdf': [6, 8]
    }

    total_facts = 0
    total_failures = 0

    for fname in pdf_files:
        fpath = os.path.join(args.data_dir, fname)
        if args.fast_demo and fname in demo_specs:
            target_pages = demo_specs[fname]
            print(f"\n--- Ingesting {fname} (Demo Pages {target_pages}) ---")
            meta, doc_facts, doc_fails = pipeline.process_pdf(fpath, target_pages=target_pages)
        else:
            print(f"\n--- Ingesting {fname} (Generic Salient Pages) ---")
            meta, doc_facts, doc_fails = pipeline.process_pdf(fpath, max_salient_pages=8)

        print(f"  Processed {meta.processed_pages} pages -> {len(doc_facts)} facts ({meta.grounded_facts_count} grounded), {len(doc_fails)} failures")
        total_facts += len(doc_facts)
        total_failures += len(doc_fails)

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
