"""
PDF metadata extraction module.

This module provides functions for extracting metadata from PDF documents.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import PyPDF2

logger = logging.getLogger(__name__)


def extract_basic_metadata(pdf_path: Union[str, Path]) -> Dict[str, str]:
    """
    Extract basic metadata from a PDF file using PyPDF2.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict[str, str]: Dictionary containing metadata fields.
    """
    logger.info(f"Extracting basic metadata from {pdf_path}")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    metadata = {}
    
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract document info
            if reader.metadata:
                for key, value in reader.metadata.items():
                    # Remove the leading slash from keys
                    clean_key = key[1:] if key.startswith("/") else key
                    metadata[clean_key] = str(value)
            
            # Add basic document properties
            metadata["PageCount"] = len(reader.pages)
            metadata["FileName"] = pdf_path.name
            metadata["FileSize"] = pdf_path.stat().st_size
            
            # Extract creation and modification dates if available
            if "CreationDate" in metadata:
                metadata["CreationDate"] = metadata["CreationDate"]
            if "ModDate" in metadata:
                metadata["ModificationDate"] = metadata["ModDate"]
    
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
        raise
    
    return metadata


def extract_toc(pdf_path: Union[str, Path]) -> List[Dict[str, Union[str, int]]]:
    """
    Extract table of contents (bookmarks) from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List[Dict[str, Union[str, int]]]: List of TOC entries with title and page number.
    """
    logger.info(f"Extracting table of contents from {pdf_path}")
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    toc_entries = []
    
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract outlines (bookmarks)
            # Note: This is a simplified implementation
            # In a real implementation, you would recursively extract nested bookmarks
            
            # PyPDF2 doesn't have great support for outlines
            # This is a placeholder for a more sophisticated implementation
            logger.warning("TOC extraction is limited in the current implementation")
            
            # As a fallback, try to extract headings from text
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Look for potential headings (simplified approach)
                    lines = text.split('\n')
                    for line in lines:
                        # Simple heuristic: short lines with all caps or ending with numbers
                        # might be headings
                        if (len(line) < 100 and (line.isupper() or 
                                                re.match(r'^[A-Z].*\d+$', line))):
                            toc_entries.append({
                                "title": line.strip(),
                                "page": i + 1
                            })
    
    except Exception as e:
        logger.error(f"Error extracting TOC from {pdf_path}: {str(e)}")
        raise
    
    return toc_entries


def extract_sections(pdf_path: Union[str, Path]) -> List[Dict[str, Union[str, int, int]]]:
    """
    Extract sections and their page ranges from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List[Dict[str, Union[str, int, int]]]: List of sections with title, start page, and end page.
    """
    logger.info(f"Extracting sections from {pdf_path}")
    
    # First get the TOC entries
    toc_entries = extract_toc(pdf_path)
    
    # Convert TOC entries to sections with page ranges
    sections = []
    for i, entry in enumerate(toc_entries):
        # Determine end page (either next section start - 1 or last page)
        end_page = toc_entries[i+1]["page"] - 1 if i < len(toc_entries) - 1 else None
        
        sections.append({
            "title": entry["title"],
            "start_page": entry["page"],
            "end_page": end_page
        })
    
    return sections


def extract_document_structure(
    pdf_path: Union[str, Path],
    text_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Union[Dict, List]]:
    """
    Extract document structure including metadata, TOC, and sections.
    Optionally links to extracted text chunks.

    Args:
        pdf_path: Path to the PDF file.
        text_dir: Optional directory containing extracted text chunks.

    Returns:
        Dict: Document structure with metadata, TOC, and sections.
    """
    logger.info(f"Extracting document structure from {pdf_path}")
    
    pdf_path = Path(pdf_path)
    
    # Extract metadata
    metadata = extract_basic_metadata(pdf_path)
    
    # Extract TOC
    toc = extract_toc(pdf_path)
    
    # Extract sections
    sections = extract_sections(pdf_path)
    
    # Create document structure
    document_structure = {
        "metadata": metadata,
        "toc": toc,
        "sections": sections
    }
    
    # Link to extracted text chunks if available
    if text_dir:
        text_dir = Path(text_dir)
        chunks_dir = text_dir / f"{pdf_path.stem}_chunks"
        
        if chunks_dir.exists():
            chunk_files = sorted(chunks_dir.glob("chunk_*.txt"))
            
            # Map chunks to sections based on content
            section_chunks = {}
            
            for section in sections:
                section_chunks[section["title"]] = []
            
            # This is a simplified approach
            # In a real implementation, you would use more sophisticated
            # methods to map chunks to sections
            for chunk_file in chunk_files:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_text = f.read()
                
                # Assign chunk to section based on content similarity
                # Here we're just checking if section title appears in chunk
                for section in sections:
                    if section["title"].lower() in chunk_text.lower():
                        section_chunks[section["title"]].append(chunk_file.name)
                        break
            
            document_structure["section_chunks"] = section_chunks
    
    return document_structure


def save_metadata(
    pdf_path: Union[str, Path],
    output_dir: Union[str, Path],
    include_structure: bool = True,
    text_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Extract and save metadata from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save the metadata.
        include_structure: Whether to include document structure.
        text_dir: Optional directory containing extracted text chunks.

    Returns:
        str: Path to the saved metadata file.
    """
    import json
    
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metadata
    metadata = extract_basic_metadata(pdf_path)
    
    # Extract document structure if requested
    if include_structure:
        document_structure = extract_document_structure(pdf_path, text_dir)
        metadata.update(document_structure)
    
    # Save metadata to file
    output_file = output_dir / f"{pdf_path.stem}_metadata.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {output_file}")
    
    return str(output_file)
