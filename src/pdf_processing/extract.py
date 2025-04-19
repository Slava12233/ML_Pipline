"""
PDF text extraction module.

This module provides functions for extracting text from PDF documents using different methods.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import PyPDF2

from src.pdf_processing import metadata

logger = logging.getLogger(__name__)


def extract_text_with_pypdf2(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from a PDF file using PyPDF2.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    logger.info(f"Extracting text from {pdf_path} using PyPDF2")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            logger.info(f"PDF has {num_pages} pages")
            
            for i, page in enumerate(reader.pages):
                logger.debug(f"Processing page {i+1}/{num_pages}")
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                else:
                    logger.warning(f"No text extracted from page {i+1}")
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise
    
    return text


def extract_text_with_document_ai(pdf_path: Union[str, Path], config: Dict) -> str:
    """
    Extract text from a PDF file using Google Document AI.

    Args:
        pdf_path: Path to the PDF file.
        config: Configuration dictionary with Document AI settings.

    Returns:
        str: Extracted text from the PDF.
    """
    logger.info(f"Extracting text from {pdf_path} using Document AI")
    
    # This is a placeholder for the actual Document AI implementation
    # In a real implementation, you would use the Google Cloud Document AI client
    
    try:
        from google.cloud import documentai_v1 as documentai
        
        # Get processor ID from config
        processor_id = config.get("document_ai", {}).get("processor_id")
        if not processor_id:
            raise ValueError("Document AI processor ID not found in config")
        
        # TODO: Implement actual Document AI processing
        # For now, fall back to PyPDF2
        logger.warning("Document AI implementation is a placeholder. Using PyPDF2 instead.")
        return extract_text_with_pypdf2(pdf_path)
        
    except ImportError:
        logger.error("google-cloud-documentai package not installed. Falling back to PyPDF2.")
        return extract_text_with_pypdf2(pdf_path)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks of specified size with overlap.

    Args:
        text: Text to split into chunks.
        chunk_size: Maximum size of each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    if not text:
        return []
    
    # Simple chunking by character count
    # In a real implementation, you would use more sophisticated methods
    # such as splitting by sentences or paragraphs
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a good breaking point (newline or space)
        if end < len(text):
            # Look for newline
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos + 1
            else:
                # Look for space
                space_pos = text.rfind(" ", start, end)
                if space_pos > start:
                    end = space_pos + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    
    return chunks


def process_pdf(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "pypdf2",
    config: Optional[Dict] = None,
    extract_meta: bool = True,
) -> Dict[str, str]:
    """
    Process a PDF file and save the extracted text and metadata.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save the extracted text.
        method: Extraction method to use ('pypdf2' or 'document_ai').
        config: Configuration dictionary.
        extract_meta: Whether to extract and save metadata.

    Returns:
        Dict[str, str]: Dictionary with paths to the saved files.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default config if none provided
    if config is None:
        config = {
            "chunk_size": 1000,
            "overlap": 200,
            "min_chunk_length": 100,
        }
    
    # Extract text using the specified method
    if method.lower() == "document_ai":
        text = extract_text_with_document_ai(pdf_path, config)
    else:
        text = extract_text_with_pypdf2(pdf_path)
    
    # Save the full text
    output_file = output_dir / f"{pdf_path.stem}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    
    # Chunk the text and save chunks
    chunks = chunk_text(
        text,
        chunk_size=config.get("chunk_size", 1000),
        overlap=config.get("overlap", 200),
    )
    
    # Filter out chunks that are too short
    min_length = config.get("min_chunk_length", 100)
    chunks = [chunk for chunk in chunks if len(chunk) >= min_length]
    
    # Save chunks
    chunks_dir = output_dir / f"{pdf_path.stem}_chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
        with open(chunk_file, "w", encoding="utf-8") as f:
            f.write(chunk)
    
    logger.info(f"Processed {pdf_path.name}: {len(text)} chars, {len(chunks)} chunks")
    
    result = {"text_file": str(output_file)}
    
    # Extract and save metadata if requested
    if extract_meta:
        try:
            # Create metadata directory
            meta_dir = output_dir / "metadata"
            meta_dir.mkdir(exist_ok=True)
            
            # Extract and save metadata
            meta_file = metadata.save_metadata(
                pdf_path=pdf_path,
                output_dir=meta_dir,
                include_structure=True,
                text_dir=output_dir,
            )
            
            result["metadata_file"] = meta_file
            logger.info(f"Extracted metadata from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            # Continue processing even if metadata extraction fails
    
    return result


def process_directory(
    pdf_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "pypdf2",
    config: Optional[Dict] = None,
    extract_meta: bool = True,
) -> Dict[str, List[str]]:
    """
    Process all PDF files in a directory.

    Args:
        pdf_dir: Directory containing PDF files.
        output_dir: Directory to save the extracted text.
        method: Extraction method to use ('pypdf2' or 'document_ai').
        config: Configuration dictionary.
        extract_meta: Whether to extract and save metadata.

    Returns:
        Dict[str, List[str]]: Dictionary with lists of paths to the saved files.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    result = {
        "text_files": [],
        "metadata_files": []
    }
    
    for pdf_file in pdf_files:
        try:
            output = process_pdf(pdf_file, output_dir, method, config, extract_meta)
            result["text_files"].append(output["text_file"])
            if "metadata_file" in output:
                result["metadata_files"].append(output["metadata_file"])
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
    
    return result
