"""
Cross-document candidate matching engine.
Pairs facts across different documents based on content-derived comparison keys,
semantic predicate clusters, and compatible value types, avoiding O(N^2) LLM comparisons.
"""

import re
from typing import List, Tuple, Set
from src.models import Fact


def canonicalize_predicate(pred: str) -> str:
    """
    Map dynamic predicate names into broad generic semantic clusters:
    e.g., 'revenue_from_contracts_with_customers' and 'turnover' -> 'revenue'
    'pin_code_reach' and 'postal_pin_codes' -> 'pin_codes'
    Works across corporate, macroeconomic, and arbitrary operational domains.
    """
    p = pred.lower().replace("-", "_").replace(" ", "_")
    p = re.sub(r"[^a-z0-9_]", "", p)

    # Financial: Revenue, turnover, sales, top-line
    if any(k in p for k in ["revenue", "turnover", "sales_income", "gross_receipts", "topline"]):
        return "revenue"
    
    # Financial: Profitability, EBITDA, margins, net income
    if any(k in p for k in ["ebitda", "operating_profit", "net_profit", "net_income"]):
        return "profitability"

    # Coverage, postal reach, geography
    if "pin" in p and ("code" in p or "reach" in p or "cover" in p):
        return "pin_codes"
    
    # Workforce, headcount, staffing, employees
    if any(k in p for k in ["workforce", "team_size", "headcount", "employee_strength", "staff_count"]):
        return "workforce"

    # Customer / Client base counts
    if "customer" in p and any(k in p for k in ["active", "count", "base", "total"]):
        return "active_customers"

    # Registered legal office address
    if "registered" in p and ("office" in p or "address" in p):
        return "registered_office"

    # Corporate / Headquarters / Operational office address
    if any(k in p for k in ["corporate", "headquarter", "hq"]) and ("office" in p or "address" in p):
        return "corporate_office"

    # General physical / facility address
    if any(k in p for k in ["address", "office"]):
        return "office_address"

    # Governance, directorship, board appointments
    if any(k in p for k in ["director", "board_member", "directorship", "trustee"]):
        return "governance_directorship"

    # Macroeconomic: GDP growth
    if "gdp" in p and any(k in p for k in ["growth", "rate", "projection", "real"]):
        return "gdp_growth"

    # Macroeconomic: Inflation, price index
    if any(k in p for k in ["inflation", "cpi", "wpi", "deflator"]):
        return "inflation"

    # Generic operational volume / shipments / freight
    if any(k in p for k in ["volume", "shipment", "tonnage", "freight"]):
        # Group by the specific volume type tokens
        tokens = [t for t in p.split("_") if t not in ["count", "number", "total", "annual", "quarterly"]]
        return "_".join(tokens[:2]) if len(tokens) >= 2 else p

    return p


def build_comparison_key(fact: Fact) -> str:
    """
    Build content-derived comparison key:
    normalized_subject_cluster | canonical_predicate_cluster
    """
    # Normalize subject (e.g. 'Delhivery Limited' -> 'delhivery')
    subj = fact.subject.lower()
    for drop in ["limited", "ltd", "corporation", "inc", "private", "pvt", "the"]:
        subj = re.sub(rf"\b{drop}\b", "", subj)
    subj = re.sub(r"[^a-z0-9]", "", subj)
    
    canon_pred = canonicalize_predicate(fact.predicate)
    return f"{subj}|{canon_pred}"


class CandidateMatcher:
    """Intelligently identifies pairs of facts across documents that warrant comparison."""

    @staticmethod
    def find_candidate_pairs(facts: List[Fact]) -> List[Tuple[Fact, Fact]]:
        """
        Find candidate pairs across DIFFERENT documents.
        Strictly prevents pairing facts from the exact same source document.
        """
        candidate_pairs: List[Tuple[Fact, Fact]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        # Group facts by comparison key
        key_buckets: dict[str, List[Fact]] = {}
        for f in facts:
            # Refresh comparison key if needed
            comp_key = build_comparison_key(f)
            key_buckets.setdefault(comp_key, []).append(f)

        def compatible(fact_a: Fact, fact_b: Fact) -> bool:
            if fact_a.value_type == fact_b.value_type or "unknown" in {fact_a.value_type, fact_b.value_type}:
                return True
            return {fact_a.value_type, fact_b.value_type}.issubset({"numeric", "currency"})

        def numeric_distance(fact_a: Fact, fact_b: Fact) -> float:
            if isinstance(fact_a.normalized_value, (int, float)) and isinstance(fact_b.normalized_value, (int, float)):
                return abs(fact_a.normalized_value - fact_b.normalized_value) / max(
                    abs(fact_a.normalized_value), abs(fact_b.normalized_value), 1.0
                )
            return 0.0

        def add_pair(fact_a: Fact, fact_b: Fact) -> None:
            if not compatible(fact_a, fact_b):
                return
            pair_id = tuple(sorted([fact_a.id, fact_b.id]))
            if pair_id not in seen_pairs:
                seen_pairs.add(pair_id)
                candidate_pairs.append((fact_a, fact_b))

        for group in key_buckets.values():
            by_document: dict[str, List[Fact]] = {}
            for fact in group:
                by_document.setdefault(fact.source_document_id, []).append(fact)

            document_ids = list(by_document)
            for i, doc_a in enumerate(document_ids):
                for doc_b in document_ids[i + 1:]:
                    facts_a, facts_b = by_document[doc_a], by_document[doc_b]
                    shared_predicates = {fact.predicate for fact in facts_a} & {fact.predicate for fact in facts_b}

                    # Prefer exact dynamic predicates. When extraction produces
                    # duplicates for one metric, retain the closest numeric
                    # counterpart instead of emitting every Cartesian pairing.
                    if shared_predicates:
                        for predicate in shared_predicates:
                            exact_pairs = [
                                (fact_a, fact_b)
                                for fact_a in facts_a
                                for fact_b in facts_b
                                if fact_a.predicate == predicate
                                and fact_b.predicate == predicate
                                and compatible(fact_a, fact_b)
                            ]
                            if exact_pairs:
                                add_pair(*min(exact_pairs, key=lambda pair: numeric_distance(*pair)))
                        continue

                    for fact_a in facts_a:
                        for fact_b in facts_b:
                            add_pair(fact_a, fact_b)

        return candidate_pairs
