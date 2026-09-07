"""
Generalization Smoke Test:
Runs ONE document from the India Macroeconomy starter dataset through the exact
generic path used for arbitrary Streamlit uploads (no filename rules, no page specs).
Validates:
1. Processing succeeds without error
2. Useful macroeconomic facts are extracted
3. Source document and page number are preserved
4. Verbatim evidence grounding works as expected
"""

import sys
import os

# Ensure UTF-8 console output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import FactKnowledgePipeline

def run_generalization_test():
    test_db = "data/test_generalization.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    pdf_path = "starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: Document not found: {pdf_path}")
        return False

    print("==================================================")
    print("Running Generalization Test on Second Domain")
    print(f"Document: {pdf_path}")
    print("==================================================")

    pipeline = FactKnowledgePipeline(db_path=test_db)
    
    # Generic pipeline path used by Streamlit uploads (salient page selection)
    meta, facts, failures = pipeline.process_pdf(pdf_path, max_salient_pages=3)

    print(f"\n[Result] Status: {meta.status}")
    print(f"[Result] Total Document Pages: {meta.page_count}, Processed Pages: {meta.processed_pages}")
    print(f"[Result] Total Facts Extracted: {len(facts)}")
    grounded_facts = [f for f in facts if f.is_grounded]
    print(f"[Result] Grounded Facts: {len(grounded_facts)} / {len(facts)} ({len(grounded_facts)/max(len(facts),1)*100:.1f}%)")
    print(f"[Result] Processing Failures / Warnings: {len(failures)}")

    print("\n--- Sample Extracted & Grounded Facts ---")
    for idx, f in enumerate(facts[:5], 1):
        print(f"{idx}. [{f.source_filename} p.{f.page_number}] {f.subject} ➔ {f.predicate}: {f.raw_value}")
        print(f"   Normalized: {f.normalized_value} {f.unit or ''}")
        print(f"   Grounding: Verified={f.is_grounded} (Score: {f.grounding_score})")
        print(f"   Evidence: \"{f.evidence_quote[:90]}...\"\n")

    # Clean up test DB
    if os.path.exists(test_db):
        os.remove(test_db)

    assert len(facts) > 0, "No facts extracted from generalization document"
    assert len(grounded_facts) > 0, "No grounded facts from generalization document"
    assert all(f.source_filename == os.path.basename(pdf_path) for f in facts), "Source filename not preserved"
    assert all(f.page_number > 0 for f in facts), "Page number not preserved"

    print("SUCCESS: Generalization smoke test passed on India Macroeconomy starter dataset.")
    return True

if __name__ == "__main__":
    success = run_generalization_test()
    if not success:
        sys.exit(1)
