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
