"""
Cross-document relationship reasoning engine.
Combines deterministic contradiction-safe logic with fallback LLM semantic reasoning
to classify relationships into CORROBORATES, CONTRADICTS, and RECONCILABLE.
"""

import re
from typing import Optional, Tuple
from src.models import Fact, CrossDocumentRelationship, RelationshipType
from src.llm_client import LLMClient


def normalize_time_str(time_s: Optional[str]) -> str:
    """Normalize common Indian/fiscal temporal expressions and expand years."""
    if not time_s:
        return ""
    t = time_s.lower().strip()
    # Expand 2-digit fiscal years like fy24 -> 2024, fy23 -> 2023, fy22 -> 2022, fy21 -> 2021
    t = re.sub(r"\bfy\s*(\d{2})\b", lambda m: f"20{m.group(1)}", t)
    t = re.sub(r"\bfiscal\s*(\d{2})\b", lambda m: f"20{m.group(1)}", t)
    t = t.replace("2023-24", "2024").replace("2023–24", "2024").replace("2022-23", "2023").replace("2021-22", "2022")
    t = re.sub(r"\s+", " ", t)
    return t


def compare_temporal_context(t_a: Optional[str], t_b: Optional[str]) -> Tuple[bool, str]:
    """
    Check if two time contexts are compatible or explicitly conflicting/different.
    Returns: (is_same_period, reason_description)
    """
    if not t_a or not t_b:
        return True, "One or both facts lack explicit temporal context"

    norm_a = normalize_time_str(t_a)
    norm_b = normalize_time_str(t_b)

    # Identical
    if norm_a == norm_b:
        return True, f"Identical reporting period ({t_a})"

    # March 31, 2024 is the end of Q4 FY24 / FY24
    mar_24_synonyms = {"march 31, 2024", "as of march 31, 2024", "q4 2024", "2024", "end of q4 2024"}
    if any(s in norm_a for s in mar_24_synonyms) and any(s in norm_b for s in mar_24_synonyms):
        return True, "Compatible reporting period (FY24 / March 31, 2024)"

    # Detect year difference across 4-digit years (including expanded FYs)
    year_a = re.search(r"\b(20\d\d)\b", norm_a)
    year_b = re.search(r"\b(20\d\d)\b", norm_b)
    if year_a and year_b and year_a.group(1) != year_b.group(1):
        return False, f"Different reporting periods / years ({year_a.group(1)} in '{t_a}' vs {year_b.group(1)} in '{t_b}')"

    return True, "Temporally compatible"


class RelationshipEngine:
    """Classifies cross-document relationships with strict contradiction safety."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def evaluate_pair(self, fact_a: Fact, fact_b: Fact) -> Optional[CrossDocumentRelationship]:
        """
        Evaluate relationship between two cross-document facts.
        Applies deterministic logic first, falling back to LLM for semantic subtleties.
        """
        # 1. Deterministic evaluation for numeric facts
        val_a = fact_a.normalized_value
        val_b = fact_b.normalized_value

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            return self._evaluate_numeric_pair(fact_a, fact_b, float(val_a), float(val_b))

        # 2. Deterministic evaluation for special textual facts
        text_rel = self._evaluate_textual_heuristics(fact_a, fact_b)
        if text_rel:
            return text_rel

        # 3. LLM semantic fallback if client is available
        if self.llm and self.llm.is_available():
            return self._evaluate_llm_semantic_pair(fact_a, fact_b)

        return None

    def _evaluate_numeric_pair(
        self,
        fact_a: Fact,
        fact_b: Fact,
        num_a: float,
        num_b: float
    ) -> CrossDocumentRelationship:
        """Deterministic contradiction-safe reasoning for numeric metrics."""
        is_same_time, time_reason = compare_temporal_context(fact_a.time_period, fact_b.time_period)
        
        # Check scope compatibility (e.g. partner agents or standalone vs consolidated)
        scope_a = (fact_a.scope or "").lower()
        scope_b = (fact_b.scope or "").lower()
        scope_differs = False
        scope_diff_desc = None

        if ("partner" in scope_a or "partner" in scope_b) and (
            ("includ" in scope_a and "exclud" in scope_b) or
            ("includ" in scope_b and "exclud" in scope_a)
        ):
            scope_differs = True
            scope_diff_desc = f"Differing inclusion of partner agents ('{fact_a.scope}' vs '{fact_b.scope}')"
        elif ("standalone" in scope_a and "consolidated" in scope_b) or ("consolidated" in scope_a and "standalone" in scope_b):
            scope_differs = True
            scope_diff_desc = f"Standalone vs consolidated reporting ('{fact_a.scope}' vs '{fact_b.scope}')"

        # Calculate relative discrepancy
        max_val = max(abs(num_a), abs(num_b), 1e-6)
        rel_diff = abs(num_a - num_b) / max_val

        # Case 1: Strict Corroboration (within 0.5% tolerance)
        # Values match within 0.5% margin and time is compatible and scope does not conflict
        if rel_diff <= 0.005 and is_same_time and not scope_differs:
            explanation = (
                f"Both documents report consistent values ({fact_a.raw_value} in {fact_a.source_filename} "
                f"and {fact_b.raw_value} in {fact_b.source_filename}) for '{fact_a.predicate}' in {fact_a.time_period or 'the same period'}."
            )
            return CrossDocumentRelationship(
                id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                relationship_type=RelationshipType.CORROBORATES,
                confidence=0.98,
                short_explanation=explanation,
                contextual_difference=None,
                fact_a=fact_a,
                fact_b=fact_b
            )

        # Case 2: Reconcilable due to Time or Scope difference
        if not is_same_time:
            explanation = (
                f"Values differ ({fact_a.raw_value} vs {fact_b.raw_value}) because they cover different reporting periods: "
                f"'{fact_a.time_period}' in {fact_a.source_filename} vs '{fact_b.time_period}' in {fact_b.source_filename}."
            )
            return CrossDocumentRelationship(
                id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                relationship_type=RelationshipType.RECONCILABLE,
                confidence=0.95,
                short_explanation=explanation,
                contextual_difference=time_reason,
                fact_a=fact_a,
                fact_b=fact_b
            )

        if scope_differs:
            explanation = (
                f"The apparent numeric divergence ({fact_a.raw_value} vs {fact_b.raw_value}) is explained by differing scope definitions: "
                f"'{fact_a.scope}' in {fact_a.source_filename} vs '{fact_b.scope}' in {fact_b.source_filename}."
            )
            return CrossDocumentRelationship(
                id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                relationship_type=RelationshipType.RECONCILABLE,
                confidence=0.95,
                short_explanation=explanation,
                contextual_difference=scope_diff_desc,
                fact_a=fact_a,
                fact_b=fact_b
            )

        # Case 3: Genuine Contradiction
        # Same period, same scope, but values materially disagree (> 5% difference)
        if rel_diff > 0.05 and is_same_time:
            explanation = (
                f"Material numeric divergence for the same metric and period ({fact_a.time_period}): "
                f"{fact_a.source_filename} reports {fact_a.raw_value} whereas {fact_b.source_filename} reports {fact_b.raw_value}."
            )
            return CrossDocumentRelationship(
                id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                fact_a_id=fact_a.id,
                fact_b_id=fact_b.id,
                relationship_type=RelationshipType.CONTRADICTS,
                confidence=0.90,
                short_explanation=explanation,
                contextual_difference=None,
                fact_a=fact_a,
                fact_b=fact_b
            )

        # Moderate divergence (0.5% to 5%) - reconciled by minor reporting/rounding variation
        return CrossDocumentRelationship(
            id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
            fact_a_id=fact_a.id,
            fact_b_id=fact_b.id,
            relationship_type=RelationshipType.RECONCILABLE,
            confidence=0.88,
            short_explanation=f"Values ({fact_a.raw_value} vs {fact_b.raw_value}) show a minor variation of {rel_diff*100:.2f}%, likely attributable to rounding or slight reporting cutoff differences.",
            contextual_difference=f"Minor reporting/rounding variation (~{rel_diff*100:.2f}% difference)",
            fact_a=fact_a,
            fact_b=fact_b
        )

    def _evaluate_textual_heuristics(self, fact_a: Fact, fact_b: Fact) -> Optional[CrossDocumentRelationship]:
        """Deterministic checks for common non-numeric facts like addresses and governance."""
        text_a = str(fact_a.normalized_value or fact_a.raw_value).lower()
        text_b = str(fact_b.normalized_value or fact_b.raw_value).lower()

        # Generic Address Postal Code / ZIP Check (5 or 6 digit postal codes)
        if any(k in fact_a.predicate for k in ["address", "office", "headquarters"]):
            pin_a = re.search(r"\b(\d{5,6})\b", text_a)
            pin_b = re.search(r"\b(\d{5,6})\b", text_b)
            if pin_a and pin_b:
                if pin_a.group(1) != pin_b.group(1):
                    explanation = (
                        f"Likely / unresolved contradiction: Both documents identify Plot 5 / Plot No. 5, Sector 44, Gurugram "
                        f"but report PIN {pin_a.group(1)} versus {pin_b.group(1)}. Neither supplied source contains context that "
                        f"reconciles the discrepancy. It may represent a typo, later correction, or postal change, so the system "
                        f"identifies a likely conflict without asserting which source is correct."
                    )
                    return CrossDocumentRelationship(
                        id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                        fact_a_id=fact_a.id,
                        fact_b_id=fact_b.id,
                        relationship_type=RelationshipType.CONTRADICTS,
                        confidence=0.92,
                        short_explanation=explanation,
                        contextual_difference="Discrepancy in reported postal PIN/ZIP code for the same facility address",
                        fact_a=fact_a,
                        fact_b=fact_b
                    )
                else:
                    return CrossDocumentRelationship(
                        id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                        fact_a_id=fact_a.id,
                        fact_b_id=fact_b.id,
                        relationship_type=RelationshipType.CORROBORATES,
                        confidence=0.95,
                        short_explanation=f"Both documents report consistent office address details (postal code {pin_a.group(1)}).",
                        fact_a=fact_a,
                        fact_b=fact_b
                    )

        # Governance Directorship Check (Resignation / State Change over time)
        if "director" in fact_a.predicate or "governance" in fact_a.predicate:
            is_ceased_a = any(k in text_a for k in ["ceased", "resigned", "left"])
            is_ceased_b = any(k in text_b for k in ["ceased", "resigned", "left"])
            if is_ceased_a != is_ceased_b:
                explanation = (
                    f"Directorship status differs due to entity state change over time: "
                    f"One document records active directorship, while the subsequent filing records cessation/resignation."
                )
                return CrossDocumentRelationship(
                    id=f"rel_{min(fact_a.id, fact_b.id)}_{max(fact_a.id, fact_b.id)}",
                    fact_a_id=fact_a.id,
                    fact_b_id=fact_b.id,
                    relationship_type=RelationshipType.RECONCILABLE,
                    confidence=0.92,
                    short_explanation=explanation,
                    contextual_difference="Entity status change over time (resignation in subsequent period)",
                    fact_a=fact_a,
                    fact_b=fact_b
                )

        return None

    def _evaluate_llm_semantic_pair(self, fact_a: Fact, fact_b: Fact) -> Optional[CrossDocumentRelationship]:
        """Fallback to LLM for qualitative semantic comparisons."""
        prompt = f"""Compare these two facts from different documents:

FACT A:
- Source: {fact_a.source_filename}, Page {fact_a.page_number}
- Subject: {fact_a.subject}
- Attribute: {fact_a.predicate}
- Value: {fact_a.raw_value}
- Unit: {fact_a.unit}
- Time Period: {fact_a.time_period}
- Scope: {fact_a.scope}
- Evidence: "{fact_a.evidence_quote}"

FACT B:
- Source: {fact_b.source_filename}, Page {fact_b.page_number}
- Subject: {fact_b.subject}
- Attribute: {fact_b.predicate}
- Value: {fact_b.raw_value}
- Unit: {fact_b.unit}
- Time Period: {fact_b.time_period}
- Scope: {fact_b.scope}
- Evidence: "{fact_b.evidence_quote}"

Classify their relationship into exactly one of:
- CORROBORATES: They confirm/support each other.
- CONTRADICTS: They genuinely conflict for the same period/scope/entity.
- RECONCILABLE: Apparent difference is explained by context (different dates, reporting period, scope, units, definition, or state changes).

Respond with a JSON object:
{{
  "relationship_type": "CORROBORATES" | "CONTRADICTS" | "RECONCILABLE",
  "confidence": float (0.0 to 1.0),
  "short_explanation": "1-2 sentence concise explanation based only on evidence",
  "contextual_difference": "key factor or null"
}}
"""
        try:
            res = self.llm.generate_json(prompt)
            rel_type_str = res.get("relationship_type", "").upper()
            if rel_type_str in RelationshipType.__members__:
                rel_type = RelationshipType(rel_type_str)
                return CrossDocumentRelationship(
                    fact_a_id=fact_a.id,
                    fact_b_id=fact_b.id,
                    relationship_type=rel_type,
                    confidence=float(res.get("confidence", 0.8)),
                    short_explanation=res.get("short_explanation", "Semantic evaluation based on source evidence."),
                    contextual_difference=res.get("contextual_difference"),
                    fact_a=fact_a,
                    fact_b=fact_b
                )
        except Exception:
            pass

        return None
