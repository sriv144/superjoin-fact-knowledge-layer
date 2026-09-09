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
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the `.env.example` template:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```
Open `.env` and add your NVIDIA API key:
```ini
# Primary LLM Provider: NVIDIA NIM (Nemotron)
# Obtain key from: https://build.nvidia.com/
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_REQUEST_TIMEOUT_SECONDS=45

DATABASE_PATH=data/facts.db
```
*(Note: API configuration is optional for browsing the precomputed assignment demonstration with 91 facts and all four required cases, but required for processing new PDFs live).*

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

Demo: [Watch the 3-minute walkthrough](https://drive.google.com/file/d/1IzCkFQfuETEWnxvUkGPODzPD10VO4PTq/view?usp=sharing)
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
             ├── Primary LLM: NVIDIA NIM (Nemotron 3.5 Lightning)
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
             ├── CORROBORATES (strict numeric margin ≤ 0.5%, same context)
             ├── RECONCILABLE (differing time period, scope, or minor rounding variation)
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
A numerical difference alone must never be prematurely labeled a contradiction. Before classifying a relationship, the engine applies multi-stage context evaluation:
- **Strict Numerical Tolerance (0.5%)**: For identical time periods and compatible scopes, values matching within **0.5%** are classified as `CORROBORATES` (e.g., ₹8,141.538 Cr vs ₹8,142 Cr shows a 0.00567% difference, well within threshold).
- **Minor Divergences (0.5% - 5%)**: Values differing slightly (e.g., ~1-3%) without conflicting scope are classified as `RECONCILABLE` attributable to rounding or reporting cutoffs, preventing false claims of strict corroboration.
- **Temporal Context**: Normalizes 2-digit fiscal years (FY24 $\rightarrow$ 2024) and calendar dates. Differing periods (e.g. FY21 vs FY24) are classified as `RECONCILABLE`.
- **Scope Definitions**: Inspects inclusion/exclusion qualifiers (e.g. "including partner agents" vs "excluding partner agents").
- **Genuine Material Conflicts**: When entity, time period, and scope are identical but values materially diverge (>5%), the pair is flagged as `CONTRADICTS`.

#### 7. Where AI Is Used vs. Where Deterministic Logic Is Used

**Where AI Is Used**:
NVIDIA Nemotron is used for:
- Identifying meaningful numerical and semantic facts from unstructured document pages.
- Converting those discoveries into the dynamic structured `Fact` schema (`subject`, `predicate`, `raw_value`, `unit`, `time_period`, `scope`, `evidence_quote`).
- Resolving ambiguous semantic relationships where deterministic comparison is insufficient.

**Where Deterministic Logic Is Used**:
Python performs:
- PDF extraction and page/evidence provenance (PyMuPDF).
- Literal quote grounding verification against raw page text (rejects ungrounded hallucinations).
- Number parsing and unit/currency normalization (e.g. ₹ million to canonical INR crore).
- Obvious numerical comparison and date checks.
- SQLite persistence and confidence guardrails.

> **Core Philosophy**: *"Nemotron proposes interpretations; deterministic code verifies what can be verified."*

#### 8. Why SQLite and Why No Graph Database
A graph database (e.g. Neo4j) introduces heavy external processes, Docker requirements, and schema overhead without adding value for factual comparison. SQLite provides zero-dependency persistence, atomic ACID transactions, and instant startup. Extracted facts and computed relationships load instantaneously in Streamlit without re-running LLM queries.

---

## Required Cases

The following table summarizes the four assignment cases discovered, grounded, and verified by our system from the starter dataset:

| Case | Classification | Source Documents | Fact A vs Fact B | System Reasoning |
| :--- | :--- | :--- | :--- | :--- |
| **Case 1: Corroboration** | `CORROBORATES` | Annual Report FY24 (p. 36) & Earnings Presentation (p. 6) | `₹81,415.38 million` vs `₹8,142 Cr` | $81,415.38\text{ million INR} = 8,141.538\text{ crore INR}$. Both report FY24 services revenue for Delhivery Limited, matching within 0.005% rounding tolerance across different reporting units. |
| **Case 2: Contradiction** | `CONTRADICTS` *(Likely / unresolved contradiction)* | Prospectus 2022 (p. 1) & Annual Report FY24 (p. 51) | `Gurugram 122002` vs `Gurugram 122001` | **Likely / unresolved contradiction**: Both documents identify Plot 5 / Plot No. 5, Sector 44, Gurugram but report PIN 122002 versus 122001. Neither supplied source contains context that reconciles the discrepancy. It may represent a typo, later correction, or postal change, so the system identifies a likely conflict without asserting which source is correct. |
| **Case 3: Contextual Reconciliation** | `RECONCILABLE` | Annual Report FY24 (p. 2) & Earnings Presentation (p. 8) | `98,135 workforce strength` vs `63,713 team size` (as of March 31, 2024) | Reconciled by scope definition: Footnote 5 of AR includes partner agents, while Footnote 4 of EP excludes partner agents and lists them separately as 34,422. Exact math: $63,713 + 34,422 = 98,135$. |
| **Case 4: Authentic Failure Case** | `FAILURE RECORDED` | Earnings Presentation (p. 9) & Annual Report (p. 36) | Multi-column tabular flattening | Standard PDF text extraction flattens multi-column tables into vertical streams, detaching numbers from column headers. Grounding verification caught and rejected ungrounded table quote reconstructions. |

### In-Depth Failure Case Analysis (Case 4)
- **What Happened**: In dense financial comparison tables (e.g. Page 9 of the Earnings Presentation showing FY22, FY23, and FY24 revenue and shipment volumes across segments), PyMuPDF extracted strings sequentially by stream position, resulting in tokens like `"59% 63% 62% ... 7,054 7,224 8,142 FY22 FY23 FY24"`. When the model attempted to extract FY23 metrics, the lack of row-column alignment caused quote reconstruction errors.
- **Why It Happened**: Standard PDF text extraction reads layout streams and discards 2D visual bounding boxes and table borders. This was an authentic developer-observed failure encountered during initial multi-column table extraction.
- **How Handled Now**: All 91 accepted facts in the database are verified grounded with literal quotes. Unsupported candidate extractions are rejected before persistence or flagged as `UNGROUNDED` with low confidence (0.2) and recorded in the audit log.
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
- **Generalization Smoke Test**: The generic pipeline (used by arbitrary Streamlit uploads) was smoke-tested on the second starter domain (`starter-datasets/india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf`) with zero domain-specific rules, hardcoded page numbers, or preset values. The system successfully parsed the document, extracted macroeconomic growth facts (e.g. India real GDP growth rate 6.5% and projected 6.6%), preserved source document filenames and page numbers, and verified 100% evidence grounding.
- **Submission Deliverables**: All pre-computed fact extractions (91 grounded facts across 3 documents), cross-document relationships (15 total: 6 Corroborates, 3 Contradicts, 6 Reconcilable), and case demonstrations are stored in `data/facts.db` and exported as standalone JSON in `sample_output/sample_run.json`.
