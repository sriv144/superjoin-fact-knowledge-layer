"""
Structured Fact Extraction module.
Prompts the LLM for structured JSON facts, validates with Pydantic,
executes deterministic grounding verification against source page text,
applies numeric/unit normalization, and constructs content-derived comparison keys.
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from src.models import Fact, ExtractionFailureEvent
from src.llm_client import LLMClient
from src.grounding import verify_grounding
from src.normalizer import normalize_fact_values
from src.matcher import build_comparison_key


class RawFactItem(BaseModel):
    subject: str = Field(description="Dynamic entity name, e.g. Delhivery Limited, India, Reserve Bank of India")
    predicate: str = Field(description="Dynamic snake_case attribute name, e.g. revenue_from_services, pin_codes_covered, corporate_office_address")
    raw_value: str = Field(description="Exact value as written in the page text, e.g. '₹8,142 Cr', '98,135', '18,793'")
    unit: Optional[str] = Field(default=None, description="Reported unit, e.g. 'INR crore', 'INR million', 'pin codes', 'percent'")
    time_period: Optional[str] = Field(default=None, description="Specific time period or date, e.g. 'FY2023-24', 'Q4 FY24', 'as of March 31, 2024'")
    scope: Optional[str] = Field(default=None, description="Scope context, e.g. 'consolidated', 'standalone', 'includes partner agents', 'excludes partner agents'")
    qualifiers: Optional[str] = Field(default=None, description="Qualifiers like 'approximate', 'projected', 'pro forma'")
    evidence_quote: str = Field(description="Verbatim quote copied from the page text that directly proves this fact")


class ExtractionResponse(BaseModel):
    facts: List[RawFactItem] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You are an expert fact extraction engine.
Your task is to extract clear, high-value, verifiable factual claims from document pages.
Target facts include:
- Financial metrics (revenue, EBITDA, turnover, net profit/loss, capital expenditure)
- Operational metrics (volumes, shipments, tonnage, pin codes reached/serviced, facility counts)
- Organizational & personnel facts (workforce/team size, partner agents, director appointments/resignations)
- Company attributes (headquarters address, pincode, corporate identity, incorporation)
- Macroeconomic statistics (real GDP growth rate, inflation rate, deficits)

CRITICAL RULES:
1. Do not extract every generic sentence. Extract only meaningful, comparable facts.
2. The evidence_quote MUST be a verbatim or near-verbatim quote from the supplied page text.
3. Explicitly capture temporal context (time_period), scope (e.g. consolidated, includes partner agents), and units.
4. Set subject to the primary organization, person, or country the fact is about - never a metric heading such as "Revenue from services" or "EBITDA". If the page omits the entity, infer it only from the document title or filename.
5. Output MUST conform to the JSON schema with a top-level key 'facts'.
"""


class FactExtractor:
    """Extracts, grounds, normalizes, and packages facts from document pages."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_from_page(
        self,
        page_text: str,
        document_id: str,
        source_filename: str,
        page_number: int,
        filter_ungrounded: bool = False
    ) -> Tuple[List[Fact], List[ExtractionFailureEvent]]:
        """
        Extract structured facts from a single page of text.
        
        Returns:
            Tuple of (grounded_facts, failure_events)
        """
        if not page_text or len(page_text.strip()) < 50:
            return [], []

        prompt = f"""Extract high-value factual statements from the following text from document '{source_filename}', page {page_number}.

TEXT:
\"\"\"
{page_text}
\"\"\"

Respond with a JSON object with key 'facts' containing a list of objects matching:
- subject: string
- predicate: string (snake_case)
- raw_value: string
- unit: string or null
- time_period: string or null
- scope: string or null
- qualifiers: string or null
- evidence_quote: string (verbatim quote from TEXT)
"""

        try:
            resp_json = self.llm.generate_json(prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)
            raw_items = ExtractionResponse.model_validate(resp_json).facts
        except Exception as e:
            failure = ExtractionFailureEvent(
                document_filename=source_filename,
                page_number=page_number,
                failure_type="llm_extraction_error",
                what_happened=f"Failed to extract structured JSON from page: {str(e)}",
                why_it_happened="LLM output did not conform to JSON schema or API error.",
                how_handled="Recorded failure event and safely skipped malformed page.",
                future_improvement="Implement secondary smaller chunk retry and fallback grammar."
            )
            return [], [failure]

        extracted_facts: List[Fact] = []
        failure_events: List[ExtractionFailureEvent] = []

        for item in raw_items:
            # 1. Evidence Grounding Verification
            is_grounded, grounding_score, matched_note = verify_grounding(page_text, item.evidence_quote)

            # 2. Value and Unit Normalization
            norm_val, norm_unit, val_type = normalize_fact_values(
                item.raw_value,
                item.unit,
                evidence_quote=item.evidence_quote,
                source_context=page_text,
            )

            fact = Fact(
                subject=item.subject.strip(),
                predicate=item.predicate.strip().lower(),
                raw_value=item.raw_value.strip(),
                normalized_value=norm_val,
                value_type=val_type,
                unit=norm_unit,
                time_period=item.time_period.strip() if item.time_period else None,
                scope=item.scope.strip() if item.scope else None,
                qualifiers=item.qualifiers.strip() if item.qualifiers else None,
                source_document_id=document_id,
                source_filename=source_filename,
                page_number=page_number,
                evidence_quote=item.evidence_quote.strip(),
                is_grounded=is_grounded,
                grounding_score=grounding_score,
                confidence=grounding_score if is_grounded else 0.2,
                extraction_notes=None if is_grounded else f"UNGROUNDED: {matched_note}"
            )
            fact.comparison_key = build_comparison_key(fact)

            if not is_grounded:
                fail_event = ExtractionFailureEvent(
                    document_filename=source_filename,
                    page_number=page_number,
                    failure_type="ungrounded_evidence",
                    what_happened=f"Fact '{fact.predicate}: {fact.raw_value}' failed grounding.",
                    why_it_happened=f"LLM evidence quote '{item.evidence_quote[:60]}...' was not found in source text.",
                    how_handled="Flagged fact as UNGROUNDED with low confidence (0.2).",
                    future_improvement="Pass surrounding bounding box text to anchor evidence quotes."
                )
                failure_events.append(fail_event)
                if filter_ungrounded:
                    continue

            extracted_facts.append(fact)

        return extracted_facts, failure_events
