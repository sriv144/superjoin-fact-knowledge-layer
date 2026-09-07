"""
Page-aware PDF extraction module using PyMuPDF (fitz).
Extracts text page-by-page, preserves exact page numbering and boundaries,
and generates stable document IDs.
"""

import hashlib
import os
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from src.models import DocumentMetadata


class PageContent:
    """Represents text extracted from a single page of a PDF."""
    def __init__(self, page_number: int, text: str, char_count: int, warnings: Optional[List[str]] = None):
        self.page_number = page_number  # 1-indexed
        self.text = text
        self.char_count = char_count
        self.warnings = warnings or []


class PDFExtractor:
    """Handles PDF ingestion, hashing, and page-aware text extraction."""

    @staticmethod
    def compute_document_id(file_path: str) -> str:
        """Generate a stable SHA-256 hash from file contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    @staticmethod
    def extract_document(
        file_path: str,
        target_pages: Optional[List[int]] = None,
        min_char_warning_threshold: int = 50
    ) -> Tuple[DocumentMetadata, List[PageContent]]:
        """
        Extract text from a PDF file page by page.
        
        Args:
            file_path: Absolute or relative path to the PDF.
            target_pages: Optional list of 1-based page numbers to extract. If None, extracts all.
            min_char_warning_threshold: Warn if extracted text on a page has fewer characters.
            
        Returns:
            Tuple of DocumentMetadata and list of PageContent objects.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc_id = PDFExtractor.compute_document_id(file_path)
        filename = os.path.basename(file_path)
        
        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages_content: List[PageContent] = []
        doc_warnings: List[str] = []

        pages_to_process = target_pages if target_pages else range(1, total_pages + 1)

        for page_num in pages_to_process:
            if page_num < 1 or page_num > total_pages:
                continue
            
            page = doc[page_num - 1]
            raw_text = page.get_text("text") or ""
            cleaned_text = raw_text.strip()
            char_count = len(cleaned_text)
            page_warnings = []

            if char_count < min_char_warning_threshold:
                warning_msg = f"Page {page_num} produced minimal text ({char_count} chars). May be scanned, graphical, or blank."
                page_warnings.append(warning_msg)
                doc_warnings.append(warning_msg)

            pages_content.append(PageContent(
                page_number=page_num,
                text=cleaned_text,
                char_count=char_count,
                warnings=page_warnings
            ))

        doc.close()

        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            page_count=total_pages,
            processed_pages=len(pages_content),
            status="extracted",
            warnings=doc_warnings
        )

        return metadata, pages_content
