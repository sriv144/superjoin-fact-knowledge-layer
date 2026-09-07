# Superjoin VIT 2026 Engineering Intern Assignment: Fact Knowledge Layer

A lightweight, robust, and evaluator-friendly **Fact Knowledge Layer** that extracts structured facts from arbitrary PDFs, verifies evidence grounding against source documents, normalizes disparate units and currencies (e.g. ₹81,415.38 million vs ₹8,142 crore), matches cross-document claims, and classifies relationships into **CORROBORATES**, **CONTRADICTS**, and **RECONCILABLE**.

---

## Setup and Run Instructions

### Prerequisites
- **Python Version**: Python 3.11+ (Tested on Python 3.11.9)
- Operating System: Windows / macOS / Linux

### 1. Clone & Navigate
```bash
git clone <YOUR_REPOSITORY_URL>
cd Superjoin-proj
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` template:
```bash
cp .env.example .env
```
Open `.env` and configure your API key. The system supports either Google Gemini (recommended default) or any OpenAI-compatible provider:
```ini
# Primary Provider: Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Alternative Provider: OpenAI / OpenRouter / Local Ollama
# OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# OPENAI_MODEL=gpt-4o-mini

DATABASE_PATH=data/facts.db
```
*(Note: A pre-extracted sample database and `sample_output/sample_run.json` are already included, so you can test and inspect results immediately even without an active API key).*

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```
- **Expected URL**: `http://localhost:8501`

### 6. Running Tests
To run the automated pytest test suite:
```bash
python -m pytest
```

### 7. Ingesting New PDFs
- Open the Streamlit web application at `http://localhost:8501`.
- In the left sidebar under **Upload PDFs**, drag and drop any arbitrary PDF documents.
- Click **Process Uploaded PDFs**. The system extracts page text, parses facts, verifies evidence grounding against source pages, normalizes values, and computes cross-document relationships incrementally.

---

## Video Demo

Demo: <VIDEO_LINK>
*(Duration: ≤ 3 minutes demonstrating PDF ingestion, fact exploration, and the four assignment cases).*

---

## Approach

### Problem Interpretation
Disclosures, reports, and filings often present interrelated facts across multiple documents. These facts may corroborate each other despite different reporting formats, diverge due to genuine organizational discrepancies, or appear contradictory on the surface but reconcile completely when temporal context, accounting scope, or definitions are taken into account. 

Our goal is not merely to compare raw numbers, but to build a **context-aware knowledge layer** grounded strictly in verifiable source evidence.

### Architecture & Data Flow

```
                Arbitrary Uploaded PDFs
                          │
                          ▼
            [Page-Aware Text Extractor] (PyMuPDF)
            ├── Preserves 1-based page numbers
            └── Emits low-text / empty page warnings
                          │
                          ▼
             [Structured Fact Extractor]
            ├── Configurable LLM (Gemini 3.6 / OpenAI)
            └── Dynamic snake_case schemas (no fixed enums)
                          │
                          ▼
             [Evidence Grounding Validator]
            ├── Verifies quote in source page text
            └── Flags/rejects hallucinated statements
                          │
                          ▼
               [Deterministic Normalizer]
            ├── Million ↔ Crore (1 Cr = 10 Mn)
            └── Strips currencies, commas, percentages
                          │
                          ▼
             [Cross-Document Matcher]
            ├── Derived keys (subject|predicate)
            └── Strictly cross-doc (no intra-doc pairs)
                          │
                          ▼
           [Contradiction-Safe Relationship Engine]
            ├── CORROBORATES (numeric margin ≤ 2%, same context)
            ├── RECONCILABLE (differing time period, scope, or state)
            └── CONTRADICTS (same period/scope, material divergence)
                          │
                          ▼
                 [SQLite Persistence]
            ├── documents, facts, relationships, failures
            └── Supports incremental ingestion
                          │
                          ▼
                  [Streamlit Web UI]
```

### Key Engineering Decisions

#### 1. PDF Ingestion (PyMuPDF)
We process PDFs page by page rather than flattening the entire document into one monolithic string. Each page retains its original 1-based page number, character count, and file hash. This ensures that any extracted fact is attributable to an exact page number for evaluator inspection.

#### 2. Dynamic Fact Schema
We avoid fixed, hardcoded enums (e.g. `REVENUE`, `EMPLOYEES`, `SHIPMENTS`). Any uploaded PDF may introduce novel domains (macroeconomic stats, logistics, clinical metrics). The extractor generates dynamic, concise `snake_case` predicates (`revenue_from_services`, `pin_codes_covered`, `corporate_office_address`) alongside temporal context and accounting scope.

#### 3. Strict Evidence Grounding
LLMs can invent or reconstruct facts. To prevent hallucinations, every candidate fact requires an `evidence_quote`. Our grounding engine performs whitespace-normalized and footnote-resilient substring matching against the source page text. If an evidence quote cannot be verified against the claimed page, it is flagged as `UNGROUNDED` with low confidence and logged in the system audit trail.

#### 4. Deterministic Normalization
Indian financial disclosures commonly oscillate between `₹ million` and `₹ crore` ($1\text{ crore} = 10\text{ million}$). Our normalizer deterministically converts Indian currency figures into canonical `INR crore`, allowing `₹81,415.38 million` and `₹8,142 crore` to evaluate directly to identical quantities without calling an LLM. Raw strings are always preserved untouched.

#### 5. Cross-Document Matching & No Intra-Document Comparison
Comparing every fact with every other fact ($O(N^2)$ LLM calls) is wasteful and slow. We cluster facts into candidate buckets using content-derived keys (`normalized_subject|canonical_predicate`). Crucially, facts from the exact same document ID are never paired, restricting comparison strictly to cross-document disclosures.

#### 6. Contradiction Safety vs. Contextual Reconciliation
A numerical difference alone must never be prematurely labeled a contradiction. Before classifying a mismatch as `CONTRADICTS`, the relationship engine inspects:
- **Temporal Context**: Normalizes 2-digit fiscal years (FY24 $\rightarrow$ 2024) and calendar years. If periods differ (e.g. FY21 vs FY24), the pair is classified as `RECONCILABLE`.
- **Scope Definitions**: Checks inclusion/exclusion qualifiers (e.g. "including partner agents" vs "excluding partner agents").
- **Only When Context is Identical**: If time, scope, and entity are identical but values materially diverge (>5%), it is classified as `CONTRADICTS`.

#### 7. Where AI is Used vs. Deterministic Logic
- **AI (LLM)**: Structured fact extraction from unstructured page text and qualitative semantic reasoning for non-numeric claims.
- **Deterministic Python**: Text extraction, footnote cleaning, evidence quote verification, unit/currency normalization, candidate pairing, numeric tolerance checks, and temporal window comparisons.

#### 8. Why SQLite and Why No Graph Database
A graph database (e.g. Neo4j) introduces heavy external processes, Docker requirements, and schema overhead without adding value for factual comparison. SQLite provides zero-dependency persistence, atomic ACID transactions, and instant startup. Extracted facts and computed relationships load instantaneously in Streamlit without re-running LLM queries.

---

## Required Cases

The following table summarizes the four assignment cases discovered, grounded, and verified by our system from the Delhivery starter dataset:

| Case | Classification | Source Documents | Fact A vs Fact B | System Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1: Corroboration** | `CORROBORATES` | Annual Report FY24 (p. 36) & Earnings Presentation (p. 6) | `₹81,415.38 million` vs `₹8,142 Cr` | $81,415.38\text{ million INR} = 8,141.538\text{ crore INR}$. Both report FY24 services revenue for Delhivery Limited, matching within 0.005% rounding tolerance across different reporting units. |
| **Case 2: Contradiction** | `CONTRADICTS` | Prospectus 2022 (p. 1) & Annual Report FY24 (p. 51) | `Gurugram 122002` vs `Gurugram 122001` | Direct geographical contradiction in the reported postal PIN code for the exact same physical headquarters building (Plot 5 / Plot No. 5, Sector 44, Gurugram). |
| **Case 3: Contextual Reconciliation** | `RECONCILABLE` | Annual Report FY24 (p. 2) & Earnings Presentation (p. 8) | `98,135 workforce strength` vs `63,713 team size` (as of March 31, 2024) | Reconciled by scope definition: Footnote 5 of AR includes partner agents, while Footnote 4 of EP excludes partner agents and lists them separately as 34,422. Exact math: $63,713 + 34,422 = 98,135$. |
| **Case 4: Authentic Failure Case** | `FAILURE RECORDED` | Earnings Presentation (p. 9) & Annual Report (p. 36) | Multi-column tabular flattening | PDF text extraction flattens multi-column tables into vertical streams, detaching numbers from column headers. Grounding verification caught and rejected ungrounded table quote reconstructions. |

### In-Depth Failure Case Analysis (Case 4)
- **What Happened**: In dense financial comparison tables (e.g. Page 9 of the Earnings Presentation showing FY22, FY23, and FY24 revenue and shipment volumes across segments), PyMuPDF extracted strings sequentially by stream position, resulting in tokens like `"59% 63% 62% ... 7,054 7,224 8,142 FY22 FY23 FY24"`. When the model attempted to extract FY23 metrics, the lack of row-column alignment caused quote reconstruction errors.
- **Why It Happened**: Standard PDF text extraction reads layout streams and discards 2D visual bounding boxes and table borders.
- **How Handled Now**: Our strict grounding validator checks candidate quotes against source page text. If an extracted evidence quote cannot be verified as a contiguous substring in the page, it is flagged as `UNGROUNDED` and assigned low confidence (0.2).
- **Future Improvement**: Implement layout-aware bounding box parsing (using `page.get_text('words')` or heuristic table cell boundary detection) to preserve 2D grid relationships before prompting.

---

## Limitations and Next Steps

### Current Limitations
1. **Complex Multi-Column Grids**: Highly dense tables without explicit cell delimiters can lose column-header associations during basic text extraction.
2. **Scanned Images**: PDFs containing pure bitmap scans without embedded OCR text layers require an OCR engine (e.g. Tesseract). The starter datasets were digital vector PDFs, so OCR was intentionally omitted to avoid heavyweight dependencies.
3. **Entity Disambiguation**: Broad corporate alias resolution (e.g. identifying subsidiary entities vs parent entities) currently relies on content-derived subject cleaning and LLM semantic context.

### Realistic Next Steps
1. **Bounding Box Table Reconstruction**: Associate table cell text with column headers using 2D geometric coordinates prior to fact extraction.
2. **Incremental Vector Index for Large Corpora**: For thousands of documents, supplement rule-based candidate keys with embedding-based approximate nearest neighbor search.
3. **Interactive PDF Highlight Viewer**: Highlight the exact bounding box of the evidence quote inside an embedded PDF canvas in the Streamlit UI.

---

## Additional Notes

- **AI & Developer Tooling**: Built and debugged using Google Antigravity paired with Python 3.11.
- **Starter Datasets**: The primary demonstration utilizes the curated Delhivery starter dataset (`01-delhivery-prospectus-2022-excerpt.pdf`, `02-delhivery-annual-report-fy24-excerpt.pdf`, `03-delhivery-q4-fy24-earnings-presentation.pdf`).
- **Submission Deliverables**: All pre-computed fact extractions, cross-document relationships, and case demonstrations are stored in `data/facts.db` and exported as standalone JSON in `sample_output/sample_run.json`.
