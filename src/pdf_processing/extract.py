#!/usr/bin/env python3
"""
PDF text and metadata extraction module.

This module provides functionality for extracting text from PDF files using
different methods such as PyPDF2 or Document AI.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import PyPDF2

logger = logging.getLogger(__name__)

def extract_text_with_pypdf2(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text and metadata from a PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Tuple of (extracted text, metadata dictionary)
    """
    text = ""
    metadata = {}
    
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
                "producer": reader.metadata.get("/Producer", ""),
                "creation_date": str(reader.metadata.get("/CreationDate", "")),
                "modification_date": str(reader.metadata.get("/ModDate", "")),
                "pages": len(reader.pages)
            }
            
            # Extract text from each page
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                else:
                    logger.warning(f"No text extracted from page {i+1}")
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise
    
    return text, metadata

def extract_text_with_document_ai(pdf_path: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text and metadata from a PDF file using Document AI.
    
    Args:
        pdf_path: Path to the PDF file
        config: Configuration dictionary containing Document AI settings
        
    Returns:
        Tuple of (extracted text, metadata dictionary)
    """
    try:
        from google.cloud import documentai_v1 as documentai
        
        # Get Document AI processor details from config
        project_id = config.get("project_id", "")
        location = config.get("location", "us-central1")
        processor_id = config.get("processor_id", "")
        
        if not project_id or not processor_id:
            raise ValueError("Document AI project_id and processor_id must be provided in config")
        
        # Initialize Document AI client
        client = documentai.DocumentProcessorServiceClient()
        name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        
        # Read the file into memory
        with open(pdf_path, "rb") as file:
            file_content = file.read()
        
        # Configure the process request
        document = {"content": file_content, "mime_type": "application/pdf"}
        request = {"name": name, "document": document}
        
        # Process the document
        result = client.process_document(request=request)
        document = result.document
        
        # Extract text
        text = document.text
        
        # Extract metadata
        metadata = {
            "mime_type": document.mime_type,
            "pages": len(document.pages),
            "entities": [
                {
                    "type": entity.type_,
                    "text": text[entity.text_anchor.text_segments[0].start_index:entity.text_anchor.text_segments[0].end_index],
                    "confidence": entity.confidence
                }
                for entity in document.entities
            ]
        }
        
        return text, metadata
    
    except ImportError:
        logger.error("Google Cloud Document AI library not installed. Run: pip install google-cloud-documentai")
        raise
    except Exception as e:
        logger.error(f"Error extracting text with Document AI: {str(e)}")
        raise

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200, min_chunk_length: int = 100) -> List[str]:
    """
    Split text into chunks of specified size with overlap.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        overlap: Overlap between consecutive chunks
        min_chunk_length: Minimum length for a valid chunk
        
    Returns:
        List of text chunks
    """
    # Split by paragraphs first
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed the chunk size, add the current chunk to the list
        if len(current_chunk) + len(para) > chunk_size and len(current_chunk) >= min_chunk_length:
            chunks.append(current_chunk.strip())
            # Keep some overlap for context
            overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_text + "\n\n"
        
        # Add the paragraph to the current chunk
        current_chunk += para + "\n\n"
    
    # Add the last chunk if it's long enough
    if len(current_chunk) >= min_chunk_length:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_pdf(
    pdf_path: str,
    output_dir: str,
    method: str = "pypdf2",
    config: Optional[Dict[str, Any]] = None,
    extract_meta: bool = True,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_length: int = 100
) -> Dict[str, Any]:
    """
    Process a PDF file, extracting text and optionally metadata.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text and metadata
        method: Extraction method ('pypdf2' or 'document_ai')
        config: Configuration dictionary
        extract_meta: Whether to extract and save metadata
        chunk_size: Maximum size of each chunk
        overlap: Overlap between consecutive chunks
        min_chunk_length: Minimum length for a valid chunk
        
    Returns:
        Dictionary with paths to output files
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    config = config or {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text and metadata using the specified method
    logger.info(f"Extracting text from {pdf_path} using {method}")
    
    if method.lower() == "document_ai":
        text, metadata = extract_text_with_document_ai(str(pdf_path), config)
    else:  # Default to PyPDF2
        text, metadata = extract_text_with_pypdf2(str(pdf_path))
    
    # Generate base filename without extension
    base_filename = pdf_path.stem
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size, overlap, min_chunk_length)
    logger.info(f"Processed {pdf_path.name}: {len(text)} chars, {len(chunks)} chunks")
    
    # Save chunked text
    text_path = output_dir / f"{base_filename}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(chunk)
            f.write("\n\n")
    
    # Save metadata if requested
    metadata_path = None
    if extract_meta:
        try:
            metadata_path = output_dir / f"{base_filename}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
    
    # Save chunks individually if there are multiple chunks
    chunk_paths = []
    if len(chunks) > 1:
        chunk_dir = output_dir / "chunks" / base_filename
        os.makedirs(chunk_dir, exist_ok=True)
        
        for i, chunk in enumerate(chunks):
            chunk_path = chunk_dir / f"chunk_{i+1}.txt"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk)
            chunk_paths.append(str(chunk_path))
    
    return {
        "text_file": str(text_path),
        "metadata_file": str(metadata_path) if metadata_path else None,
        "chunks": chunks,
        "chunk_files": chunk_paths
    }

def process_directory(
    pdf_dir: str,
    output_dir: str,
    method: str = "pypdf2",
    config: Optional[Dict[str, Any]] = None,
    extract_meta: bool = True
) -> Dict[str, Any]:
    """
    Process all PDF files in a directory.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save extracted text and metadata
        method: Extraction method ('pypdf2' or 'document_ai')
        config: Configuration dictionary
        extract_meta: Whether to extract and save metadata
        
    Returns:
        Dictionary with lists of processed files
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    config = config or {}
    
    # Get processing parameters from config
    chunk_size = config.get("chunk_size", 1000)
    overlap = config.get("overlap", 200)
    min_chunk_length = config.get("min_chunk_length", 100)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    results = {
        "text_files": [],
        "metadata_files": [],
        "chunk_files": []
    }
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            result = process_pdf(
                pdf_path=str(pdf_file),
                output_dir=output_dir,
                method=method,
                config=config,
                extract_meta=extract_meta,
                chunk_size=chunk_size,
                overlap=overlap,
                min_chunk_length=min_chunk_length
            )
            
            results["text_files"].append(result["text_file"])
            if result["metadata_file"]:
                results["metadata_files"].append(result["metadata_file"])
            results["chunk_files"].extend(result["chunk_files"])
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
    
    return results
