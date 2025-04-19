"""
PDF processing module for fine-tuning pipeline.

This module provides functionality for extracting text and metadata from PDF files.
"""

from src.pdf_processing.extract import (
    extract_text_with_pypdf2,
    extract_text_with_document_ai,
    chunk_text,
    process_pdf,
    process_directory
)

__all__ = [
    'extract_text_with_pypdf2',
    'extract_text_with_document_ai',
    'chunk_text',
    'process_pdf',
    'process_directory'
]
