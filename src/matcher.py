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
    Map dynamic predicate names into broad semantic concept clusters:
    e.g., 'revenue_from_contracts_with_customers' and 'turnover' -> 'revenue'
    'pin_code_reach' and 'postal_pin_codes' -> 'pin_codes'
    """
    p = pred.lower().replace("-", "_").replace(" ", "_")
    p = re.sub(r"[^a-z0-9_]", "", p)

    # Financial revenue / turnover
    if any(k in p for k in ["revenue", "turnover", "income_from_operations", "sales_income"]):
        return "revenue"
    
    # Coverage / Pincodes
    if "pin" in p and ("code" in p or "reach" in p):
        return "pin_codes"
    
    # Workforce / Headcount / Team
    if any(k in p for k in ["workforce", "team_size", "headcount", "employee"]):
        return "workforce"

    # Active Customers
    if "customer" in p and ("active" in p or "count" in p):
        return "active_customers"

    # Express parcel shipments / volume
    if "express_parcel" in p:
        return "express_parcel_volume"

    # Part truckload / PTL
    if "ptl" in p or "part_truckload" in p or "parttruckload" in p:
        return "ptl_tonnage"

    # Corporate / Registered Address
    if "address" in p or "office" in p or "headquarters" in p:
        return "office_address"

    # Governance / Directors
    if "director" in p or "board" in p or "appointment" in p or "designation" in p:
        return "governance_directorship"

    # Real GDP growth
    if "gdp" in p and ("growth" in p or "rate" in p):
        return "gdp_growth"

    # Inflation
    if "inflation" in p or "cpi" in p:
        return "cpi_inflation"

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

        for comp_key, group in key_buckets.items():
            n = len(group)
            for i in range(n):
                for j in range(i + 1, n):
                    fact_a = group[i]
                    fact_b = group[j]

                    # 1. Reject intra-document comparison
                    if fact_a.source_document_id == fact_b.source_document_id:
                        continue

                    # 2. Check pair deduplication
                    pair_id = tuple(sorted([fact_a.id, fact_b.id]))
                    if pair_id in seen_pairs:
                        continue
                    seen_pairs.add(pair_id)

                    # 3. Compatible value types
                    if fact_a.value_type != fact_b.value_type and fact_a.value_type != "unknown" and fact_b.value_type != "unknown":
                        # Allow currency vs numeric (e.g. INR crore vs plain float)
                        types = {fact_a.value_type, fact_b.value_type}
                        if not types.issubset({"numeric", "currency"}):
                            continue

                    candidate_pairs.append((fact_a, fact_b))

        return candidate_pairs
