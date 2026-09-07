"""
Deterministic normalization engine for numbers, units, currencies, and percentages.
Enables cross-document comparability (e.g. ₹81,415.38 million vs ₹8,142 crore)
while strictly preserving the raw source value.
"""

import re
from typing import Tuple, Optional, Union


def parse_numeric_value(raw_val: str) -> Tuple[Optional[float], Optional[str], str]:
    """
    Parse a raw string into a normalized numeric value, normalized unit, and value_type.
    
    Handles:
    - Indian numbering: Lakh, Crore (Cr)
    - Western numbering: Million (Mn), Billion (Bn), K (thousands)
    - Currency symbols: ₹, Rs, INR, $, USD
    - Percentages: %, per cent
    - Direct integers and floats with commas
    
    Standardizes Indian financial currency figures to 'INR crore'
    so that ₹81,415.38 million and ₹8,142 crore evaluate directly to comparable numbers.
    """
    if not raw_val or not isinstance(raw_val, str):
        return None, None, "unknown"

    val_str = raw_val.strip()
    clean_str = val_str.replace(",", "").replace("\u20b9", "Rs").replace("₹", "Rs")
    
    # 1. Percentage check
    pct_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*(?:%|per\s*cent)", clean_str, re.IGNORECASE)
    if pct_match:
        num = float(pct_match.group(1).replace(" ", ""))
        return round(num, 4), "percent", "percentage"

    # 2. Currency check (INR / USD)
    is_inr = bool(re.search(r"(?:rs\.?|inr)", clean_str, re.IGNORECASE))
    is_usd = bool(re.search(r"(?:\$|usd)", clean_str, re.IGNORECASE))

    # 3. Units & Multipliers
    # Patterns for Crore / Cr
    cr_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*(?:cr(?:ore)?s?)\b", clean_str, re.IGNORECASE)
    if cr_match:
        num = float(cr_match.group(1).replace(" ", ""))
        unit = "INR crore" if is_inr or not is_usd else "USD crore"
        return round(num, 4), unit, "currency"

    # Patterns for Million / Mn (normalize INR million -> INR crore by / 10)
    mn_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*(?:mn|million)s?\b", clean_str, re.IGNORECASE)
    if mn_match:
        num = float(mn_match.group(1).replace(" ", ""))
        if is_inr:
            # 1 Crore = 10 Million -> convert to crore for unified baseline
            return round(num / 10.0, 4), "INR crore", "currency"
        elif is_usd:
            return round(num, 4), "USD million", "currency"
        else:
            # Non-currency metric (e.g. shipments, volume, tonnes)
            # Check if tonnes/parcels/sq ft mentioned
            if "ton" in clean_str.lower():
                return round(num, 4), "million tonnes", "numeric"
            elif "parcel" in clean_str.lower() or "shipment" in clean_str.lower():
                return round(num, 4), "million parcels", "numeric"
            elif "sq ft" in clean_str.lower():
                return round(num, 4), "million sq ft", "numeric"
            return round(num, 4), "million", "numeric"

    # Patterns for Lakh / Lac
    lakh_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*(?:lakh|lac)s?\b", clean_str, re.IGNORECASE)
    if lakh_match:
        num = float(lakh_match.group(1).replace(" ", ""))
        if is_inr:
            # 1 Crore = 100 Lakh -> convert to crore
            return round(num / 100.0, 4), "INR crore", "currency"
        return round(num, 4), "lakh", "numeric"

    # Patterns for Billion / Bn
    bn_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*(?:bn|billion)s?\b", clean_str, re.IGNORECASE)
    if bn_match:
        num = float(bn_match.group(1).replace(" ", ""))
        if is_inr:
            # 1 Billion = 100 Crore
            return round(num * 100.0, 4), "INR crore", "currency"
        elif is_usd:
            return round(num * 1000.0, 4), "USD million", "currency"
        return round(num, 4), "billion", "numeric"

    # Patterns for K (thousands)
    k_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)\s*(?:k|thousand)s?\b", clean_str, re.IGNORECASE)
    if k_match:
        num = float(k_match.group(1).replace(" ", ""))
        if "tonne" in clean_str.lower() or "ton" in clean_str.lower():
            # 1,429K tonnes = 1.429 million tonnes
            return round(num / 1000.0, 4), "million tonnes", "numeric"
        return round(num * 1000.0, 4), "count", "numeric"

    # Check if this is primarily a text sentence or multi-word address rather than a number
    alpha_words = re.findall(r"[a-zA-Z]{2,}", clean_str)
    has_currency = is_inr or is_usd
    if len(alpha_words) >= 3 and not has_currency:
        norm_text = re.sub(r"\s+", " ", val_str).strip()
        return None, None, "text"

    # 4. Standard plain numeric pattern (with optional sign and decimals)
    num_match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)", clean_str)
    if num_match:
        try:
            num = float(num_match.group(1).replace(" ", ""))
            if is_inr:
                unit = "INR"
                val_type = "currency"
            elif is_usd:
                unit = "USD"
                val_type = "currency"
            elif "pin" in clean_str.lower() or "code" in clean_str.lower():
                unit = "pin_codes"
                val_type = "numeric"
            else:
                unit = "count"
                val_type = "numeric"
            return round(num, 4), unit, val_type
        except ValueError:
            pass

    # 5. Non-numeric / Semantic Text value
    # Canonicalize whitespace
    norm_text = re.sub(r"\s+", " ", val_str).strip()
    return None, None, "text"


def normalize_fact_values(
    raw_value: str,
    raw_unit: Optional[str] = None
) -> Tuple[Optional[Union[float, str]], Optional[str], str]:
    """
    Given a raw value string and optional unit string, compute normalized representation.
    Returns: (normalized_value, normalized_unit, value_type)
    """
    combined = f"{raw_value} {raw_unit or ''}".strip()
    num_val, unit, val_type = parse_numeric_value(combined)
    
    if num_val is not None:
        return num_val, unit, val_type
    
    # Return cleaned string for qualitative/semantic values
    cleaned_str = re.sub(r"\s+", " ", raw_value).strip()
    return cleaned_str, raw_unit or "text", "text"
