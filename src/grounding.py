"""
Evidence grounding verification module.
Verifies that claimed LLM evidence quotes strictly exist in the source page text,
accounting for PDF line-wrapping, whitespace variations, and special typographic characters.
"""

import re
from typing import Tuple


def normalize_text_for_matching(text: str) -> str:
    """
    Standardize text to enable robust substring matching across PDF extraction artifacts:
    - Normalizes unicode quotes, hyphens, and whitespace
    - Strips soft hyphens and line-break artifacts
    """
    if not text:
        return ""
    
    # Normalize unicode quotes and dashes
    s = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("–", "-").replace("—", "-").replace("\u00ad", "")  # soft hyphen
    s = s.replace("\u20b9", "Rs")  # normalize rupee symbol for matching
    
    # Strip parenthesized footnote references common in PDF disclosures like (1), (2), (1,5)
    s = re.sub(r"\(\s*\d+(?:\s*,\s*\d+)*\s*\)", " ", s)
    
    # Replace newlines, tabs, and multiple spaces with a single space
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def verify_grounding(
    page_text: str,
    evidence_quote: str,
    min_overlap_ratio: float = 0.70
) -> Tuple[bool, float, str]:
    """
    Verify whether an evidence quote exists in the source page text.
    
    Returns:
        Tuple of:
        - is_grounded (bool): True if verified
        - score (float): 1.0 for exact/cleaned match, or overlap fraction [0.0 - 1.0]
        - matched_text (str): The verified text or diagnostic reason
    """
    if not evidence_quote or not page_text:
        return False, 0.0, "Empty quote or page text"

    quote_raw = evidence_quote.strip()
    
    # 1. Exact raw substring match
    if quote_raw in page_text:
        return True, 1.0, quote_raw

    norm_page = normalize_text_for_matching(page_text)
    norm_quote = normalize_text_for_matching(quote_raw)

    # 2. Normalized whitespace / punctuation substring match
    if norm_quote in norm_page:
        return True, 1.0, norm_quote

    # 3. Token-level windowed overlap matching
    quote_tokens = [t for t in norm_quote.split(" ") if len(t) > 1]
    if not quote_tokens:
        return False, 0.0, "Quote contains no alphanumeric tokens"

    # Count how many quote tokens appear anywhere in the normalized page
    page_tokens_set = set(norm_page.split(" "))
    matched_tokens = [t for t in quote_tokens if t in page_tokens_set]
    overlap_ratio = len(matched_tokens) / len(quote_tokens)

    # Also check if token sequence appears as a contiguous subsequence
    # (Checking 3-gram matches)
    if len(quote_tokens) >= 3:
        trigrams = [" ".join(quote_tokens[i:i+3]) for i in range(len(quote_tokens)-2)]
        matched_trigrams = [tg for tg in trigrams if tg in norm_page]
        trigram_ratio = len(matched_trigrams) / len(trigrams)
        # Combined score
        score = 0.4 * overlap_ratio + 0.6 * trigram_ratio
    else:
        score = overlap_ratio

    if score >= min_overlap_ratio:
        return True, round(score, 3), f"Partial fuzzy grounding ({round(score*100, 1)}% token overlap)"

    return False, round(score, 3), f"Ungrounded: only {round(score*100, 1)}% overlap with source page"
