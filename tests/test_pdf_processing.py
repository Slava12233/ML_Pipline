"""
Tests for the PDF processing module.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.pdf_processing import extract


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


def test_extract_text_with_pypdf2(sample_pdf_file):
    """Test extracting text from a PDF file using PyPDF2."""
    # Extract text
    text = extract.extract_text_with_pypdf2(sample_pdf_file)
    
    # Check that text was extracted
    assert text
    assert "Sample PDF content" in text


def test_chunk_text():
    """Test chunking text."""
    # Sample text
    text = "This is the first sentence. This is the second sentence. " * 20
    
    # Chunk text
    chunks = extract.chunk_text(text, chunk_size=100, overlap=20)
    
    # Check that chunks were created
    assert chunks
    assert len(chunks) > 1
    
    # Check that chunks have the expected size
    for chunk in chunks:
        assert len(chunk) <= 100
    
    # Check that chunks overlap
    for i in range(len(chunks) - 1):
        overlap_text = chunks[i][-20:]
        assert overlap_text in chunks[i+1]


def test_process_pdf(sample_pdf_file, tmp_path):
    """Test processing a PDF file."""
    # Process PDF
    output_file = extract.process_pdf(
        pdf_path=sample_pdf_file,
        output_dir=tmp_path,
        method="pypdf2",
    )
    
    # Check that output file was created
    assert os.path.exists(output_file)
    
    # Check that chunks directory was created
    chunks_dir = tmp_path / f"{sample_pdf_file.stem}_chunks"
    assert chunks_dir.exists()
    
    # Check that chunks were created
    chunks = list(chunks_dir.glob("chunk_*.txt"))
    assert chunks


def test_process_directory(tmp_path):
    """Test processing a directory of PDF files."""
    # Create PDF directory
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    
    # Create sample PDF files
    for i in range(3):
        pdf_path = pdf_dir / f"sample_{i}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>/Parent 2 0 R>>\nendobj\n4 0 obj\n<</Length 44>>stream\nBT\n/F1 12 Tf\n100 700 Td\n(Sample PDF content) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000056 00000 n\n0000000111 00000 n\n0000000254 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n345\n%%EOF")
    
    # Create output directory
    output_dir = tmp_path / "output"
    
    # Mock the process_pdf function to avoid actual processing
    with mock.patch("src.pdf_processing.extract.process_pdf") as mock_process_pdf:
        mock_process_pdf.return_value = "mocked_output_file.txt"
        
        # Process directory
        output_files = extract.process_directory(
            pdf_dir=pdf_dir,
            output_dir=output_dir,
            method="pypdf2",
        )
        
        # Check that process_pdf was called for each PDF file
        assert mock_process_pdf.call_count == 3
        
        # Check that output files were returned
        assert len(output_files) == 3


def test_extract_text_with_document_ai(sample_pdf_file):
    """Test extracting text from a PDF file using Document AI."""
    # Mock the Document AI client
    with mock.patch("google.cloud.documentai_v1") as mock_documentai:
        # Configure the mock
        mock_documentai.DocumentProcessorServiceClient.return_value = mock.MagicMock()
        
        # Extract text
        config = {"document_ai": {"processor_id": "test-processor-id"}}
        
        # This should fall back to PyPDF2 since we're not actually calling Document AI
        text = extract.extract_text_with_document_ai(sample_pdf_file, config)
        
        # Check that text was extracted
        assert text
        assert "Sample PDF content" in text


def test_process_pdf_with_invalid_path():
    """Test processing a PDF file with an invalid path."""
    with pytest.raises(FileNotFoundError):
        extract.process_pdf(
            pdf_path="nonexistent.pdf",
            output_dir=tempfile.gettempdir(),
            method="pypdf2",
        )


def test_process_directory_with_invalid_path():
    """Test processing a directory with an invalid path."""
    with pytest.raises(FileNotFoundError):
        extract.process_directory(
            pdf_dir="nonexistent_dir",
            output_dir=tempfile.gettempdir(),
            method="pypdf2",
        )
