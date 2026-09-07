"""
SQLite database layer for persisting documents, extracted facts,
cross-document relationships, and processing failure events.
Supports incremental updates and zero re-computation on reload.
"""

import sqlite3
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager
from src.models import Fact, CrossDocumentRelationship, DocumentMetadata, ExtractionFailureEvent, RelationshipType


class Database:
    def __init__(self, db_path: str = "data/facts.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    processed_pages INTEGER DEFAULT 0,
                    extracted_facts_count INTEGER DEFAULT 0,
                    grounded_facts_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    warnings TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    raw_value TEXT NOT NULL,
                    normalized_value TEXT,
                    value_type TEXT,
                    unit TEXT,
                    time_period TEXT,
                    scope TEXT,
                    qualifiers TEXT,
                    comparison_key TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    evidence_quote TEXT NOT NULL,
                    is_grounded INTEGER DEFAULT 1,
                    grounding_score REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    extraction_notes TEXT,
                    raw_data TEXT NOT NULL,
                    FOREIGN KEY (source_document_id) REFERENCES documents(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    fact_a_id TEXT NOT NULL,
                    fact_b_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    short_explanation TEXT NOT NULL,
                    contextual_difference TEXT,
                    raw_data TEXT NOT NULL,
                    FOREIGN KEY (fact_a_id) REFERENCES facts(id),
                    FOREIGN KEY (fact_b_id) REFERENCES facts(id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_events (
                    id TEXT PRIMARY KEY,
                    document_filename TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    failure_type TEXT NOT NULL,
                    what_happened TEXT NOT NULL,
                    why_it_happened TEXT NOT NULL,
                    how_handled TEXT NOT NULL,
                    future_improvement TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_document(self, doc: DocumentMetadata):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (id, filename, page_count, processed_pages, extracted_facts_count, grounded_facts_count, status, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    processed_pages = excluded.processed_pages,
                    extracted_facts_count = excluded.extracted_facts_count,
                    grounded_facts_count = excluded.grounded_facts_count,
                    status = excluded.status,
                    warnings = excluded.warnings
            """, (
                doc.id, doc.filename, doc.page_count, doc.processed_pages,
                doc.extracted_facts_count, doc.grounded_facts_count,
                doc.status, json.dumps(doc.warnings)
            ))
            conn.commit()

    def save_facts(self, facts: List[Fact]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for f in facts:
                cursor.execute("""
                    INSERT INTO facts (
                        id, subject, predicate, raw_value, normalized_value, value_type,
                        unit, time_period, scope, qualifiers, comparison_key,
                        source_document_id, source_filename, page_number,
                        evidence_quote, is_grounded, grounding_score, confidence,
                        extraction_notes, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        subject = excluded.subject,
                        predicate = excluded.predicate,
                        raw_value = excluded.raw_value,
                        normalized_value = excluded.normalized_value,
                        value_type = excluded.value_type,
                        unit = excluded.unit,
                        time_period = excluded.time_period,
                        scope = excluded.scope,
                        qualifiers = excluded.qualifiers,
                        comparison_key = excluded.comparison_key,
                        is_grounded = excluded.is_grounded,
                        grounding_score = excluded.grounding_score,
                        confidence = excluded.confidence,
                        extraction_notes = excluded.extraction_notes,
                        raw_data = excluded.raw_data
                """, (
                    f.id, f.subject, f.predicate, f.raw_value,
                    str(f.normalized_value) if f.normalized_value is not None else None,
                    f.value_type, f.unit, f.time_period, f.scope, f.qualifiers,
                    f.comparison_key, f.source_document_id, f.source_filename,
                    f.page_number, f.evidence_quote, 1 if f.is_grounded else 0,
                    f.grounding_score, f.confidence, f.extraction_notes,
                    f.model_dump_json()
                ))
            conn.commit()

    def save_relationships(self, relationships: List[CrossDocumentRelationship]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for r in relationships:
                cursor.execute("""
                    INSERT INTO relationships (
                        id, fact_a_id, fact_b_id, relationship_type, confidence,
                        short_explanation, contextual_difference, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        relationship_type = excluded.relationship_type,
                        confidence = excluded.confidence,
                        short_explanation = excluded.short_explanation,
                        contextual_difference = excluded.contextual_difference,
                        raw_data = excluded.raw_data
                """, (
                    r.id, r.fact_a_id, r.fact_b_id, r.relationship_type.value,
                    r.confidence, r.short_explanation, r.contextual_difference,
                    r.model_dump_json()
                ))
            conn.commit()

    def save_failure_event(self, event: ExtractionFailureEvent):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_events (
                    id, document_filename, page_number, failure_type,
                    what_happened, why_it_happened, how_handled, future_improvement
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    what_happened = excluded.what_happened,
                    why_it_happened = excluded.why_it_happened,
                    how_handled = excluded.how_handled,
                    future_improvement = excluded.future_improvement
            """, (
                event.id, event.document_filename, event.page_number, event.failure_type,
                event.what_happened, event.why_it_happened, event.how_handled, event.future_improvement
            ))
            conn.commit()

    def get_documents(self) -> List[DocumentMetadata]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY created_at ASC")
            rows = cursor.fetchall()
            docs = []
            for r in rows:
                docs.append(DocumentMetadata(
                    id=r["id"],
                    filename=r["filename"],
                    page_count=r["page_count"],
                    processed_pages=r["processed_pages"],
                    extracted_facts_count=r["extracted_facts_count"],
                    grounded_facts_count=r["grounded_facts_count"],
                    status=r["status"],
                    warnings=json.loads(r["warnings"]) if r["warnings"] else []
                ))
            return docs

    def get_facts(self, grounded_only: bool = False, document_id: Optional[str] = None) -> List[Fact]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT raw_data FROM facts WHERE 1=1"
            params = []
            if grounded_only:
                query += " AND is_grounded = 1"
            if document_id:
                query += " AND source_document_id = ?"
                params.append(document_id)
            query += " ORDER BY source_filename ASC, page_number ASC"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [Fact.model_validate_json(r["raw_data"]) for r in rows]

    def get_relationships(self, rel_type: Optional[RelationshipType] = None) -> List[CrossDocumentRelationship]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT raw_data FROM relationships WHERE 1=1"
            params = []
            if rel_type:
                query += " AND relationship_type = ?"
                params.append(rel_type.value)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [CrossDocumentRelationship.model_validate_json(r["raw_data"]) for r in rows]

    def get_failure_events(self) -> List[ExtractionFailureEvent]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM processing_events ORDER BY created_at DESC")
            rows = cursor.fetchall()
            events = []
            for r in rows:
                events.append(ExtractionFailureEvent(
                    id=r["id"],
                    document_filename=r["document_filename"],
                    page_number=r["page_number"],
                    failure_type=r["failure_type"],
                    what_happened=r["what_happened"],
                    why_it_happened=r["why_it_happened"],
                    how_handled=r["how_handled"],
                    future_improvement=r["future_improvement"]
                ))
            return events

    def clear_all(self):
        """Reset the database for fresh ingestion."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM relationships")
            cursor.execute("DELETE FROM facts")
            cursor.execute("DELETE FROM documents")
            cursor.execute("DELETE FROM processing_events")
            conn.commit()
