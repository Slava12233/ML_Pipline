"""
Tests for the PDF metadata extraction module.
"""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from src.pdf_processing import metadata


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>/Parent 2 0 R>>\nendobj\n4 0 obj\n<</Length 44>>stream\nBT\n/F1 12 Tf\n100 700 Td\n(Sample PDF content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000254 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n345\n%%EOF"


@pytest.fixture
def sample_pdf_file(sample_pdf_content, tmp_path):
    """Create a sample PDF file for testing."""
    pdf_path = tmp_path / "sample.pdf"
    with open(pdf_path, "wb") as f:
        f.write(sample_pdf_content)
    return pdf_path


def test_extract_basic_metadata(sample_pdf_file):
    """Test extracting basic metadata from a PDF file."""
    # Extract metadata
    meta = metadata.extract_basic_metadata(sample_pdf_file)
    
    # Check that metadata was extracted
    assert meta
    assert "PageCount" in meta
    assert meta["PageCount"] == 1
    assert "FileName" in meta
    assert meta["FileName"] == "sample.pdf"
    assert "FileSize" in meta


def test_extract_toc(sample_pdf_file):
    """Test extracting table of contents from a PDF file."""
    # Extract TOC
    toc = metadata.extract_toc(sample_pdf_file)
    
    # Check that TOC was extracted
    # Note: Our sample PDF doesn't have a TOC, so this might be empty
    # or might contain entries based on heuristics
    assert isinstance(toc, list)


def test_extract_sections(sample_pdf_file):
    """Test extracting sections from a PDF file."""
    # Extract sections
    sections = metadata.extract_sections(sample_pdf_file)
    
    # Check that sections were extracted
    assert isinstance(sections, list)
    
    # If sections were found, check their structure
    for section in sections:
        assert "title" in section
        assert "start_page" in section
        # end_page might be None for the last section
        assert "end_page" in section


def test_extract_document_structure(sample_pdf_file):
    """Test extracting document structure from a PDF file."""
    # Extract document structure
    structure = metadata.extract_document_structure(sample_pdf_file)
    
    # Check that structure was extracted
    assert structure
    assert "metadata" in structure
    assert "toc" in structure
    assert "sections" in structure


def test_save_metadata(sample_pdf_file, tmp_path):
    """Test saving metadata to a file."""
    # Save metadata
    output_file = metadata.save_metadata(
        pdf_path=sample_pdf_file,
        output_dir=tmp_path,
        include_structure=True,
    )
    
    # Check that output file was created
    assert os.path.exists(output_file)
    
    # Check that the file contains valid JSON
    with open(output_file, "r") as f:
        data = json.load(f)
    
    # Check that the JSON contains the expected fields
    assert "metadata" in data
    assert "toc" in data
    assert "sections" in data


def test_extract_document_structure_with_text_dir(sample_pdf_file, tmp_path):
    """Test extracting document structure with text directory."""
    # Create a text directory with chunks
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    
    # Create a chunks directory
    chunks_dir = text_dir / f"{sample_pdf_file.stem}_chunks"
    chunks_dir.mkdir()
    
    # Create some chunk files
    for i in range(3):
        chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
        with open(chunk_file, "w") as f:
            f.write(f"Sample chunk {i} content")
    
    # Extract document structure
    structure = metadata.extract_document_structure(
        pdf_path=sample_pdf_file,
        text_dir=text_dir,
    )
    
    # Check that structure was extracted
    assert structure
    assert "metadata" in structure
    assert "toc" in structure
    assert "sections" in structure
    
    # section_chunks might not be present if no sections were found
    # or if no chunks matched any sections


def test_extract_basic_metadata_with_invalid_path():
    """Test extracting metadata with an invalid path."""
    with pytest.raises(FileNotFoundError):
        metadata.extract_basic_metadata("nonexistent.pdf")


def test_extract_toc_with_invalid_path():
    """Test extracting TOC with an invalid path."""
    with pytest.raises(FileNotFoundError):
        metadata.extract_toc("nonexistent.pdf")


def test_extract_sections_with_invalid_path():
    """Test extracting sections with an invalid path."""
    with pytest.raises(FileNotFoundError):
        metadata.extract_sections("nonexistent.pdf")
