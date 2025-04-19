"""
Training data generation module.

This module provides functions for generating training data from extracted PDF text.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def load_text_chunks(text_dir: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Load text chunks from a directory.

    Args:
        text_dir: Directory containing extracted text chunks.

    Returns:
        Dict[str, List[str]]: Dictionary mapping document names to lists of text chunks.
    """
    text_dir = Path(text_dir)
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")
    
    chunks_by_doc = {}
    
    # Look for chunk directories
    for item in text_dir.iterdir():
        if item.is_dir() and item.name.endswith("_chunks"):
            doc_name = item.name.replace("_chunks", "")
            chunks = []
            
            # Load all chunk files
            for chunk_file in sorted(item.glob("chunk_*.txt")):
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_text = f.read().strip()
                    if chunk_text:
                        chunks.append(chunk_text)
            
            if chunks:
                chunks_by_doc[doc_name] = chunks
                logger.info(f"Loaded {len(chunks)} chunks for document {doc_name}")
    
    return chunks_by_doc


def generate_question_for_chunk(chunk: str) -> str:
    """
    Generate a question for a text chunk.

    Args:
        chunk: Text chunk to generate a question for.

    Returns:
        str: Generated question.
    """
    # This is a placeholder for a more sophisticated question generation approach
    # In a real implementation, you would use NLP techniques or an LLM to generate questions
    
    # Simple approach: extract the first sentence and turn it into a question
    first_sentence = chunk.split(".")[0].strip()
    
    # Remove common stopwords from the beginning
    stopwords = ["the", "a", "an", "in", "on", "at", "to", "for", "with", "by"]
    words = first_sentence.split()
    
    if words and words[0].lower() in stopwords:
        first_sentence = " ".join(words[1:])
    
    # Generate a simple question
    question_templates = [
        f"What can you tell me about {first_sentence}?",
        f"Explain {first_sentence} in detail.",
        f"Describe {first_sentence}.",
        f"What is {first_sentence}?",
        f"Can you provide information about {first_sentence}?",
    ]
    
    return random.choice(question_templates)


def generate_instruction_for_chunk(chunk: str) -> str:
    """
    Generate an instruction for a text chunk.

    Args:
        chunk: Text chunk to generate an instruction for.

    Returns:
        str: Generated instruction.
    """
    # This is a placeholder for a more sophisticated instruction generation approach
    # In a real implementation, you would use NLP techniques or an LLM
    
    # Simple approach: extract keywords and create an instruction
    words = chunk.split()
    if len(words) > 10:
        keywords = " ".join(words[:10])
    else:
        keywords = " ".join(words)
    
    instruction_templates = [
        f"Summarize the following information about {keywords}...",
        f"Explain the concept of {keywords} based on the following text...",
        f"Based on the documentation, describe {keywords}...",
        f"Using the following information, explain {keywords}...",
        f"Provide a detailed explanation of {keywords} from the documentation...",
    ]
    
    return random.choice(instruction_templates)


def create_qa_pair(chunk: str) -> Dict[str, str]:
    """
    Create a question-answer pair from a text chunk.

    Args:
        chunk: Text chunk to create a QA pair from.

    Returns:
        Dict[str, str]: Dictionary with input_text (question) and output_text (answer).
    """
    question = generate_question_for_chunk(chunk)
    
    return {
        "input_text": question,
        "output_text": chunk,
    }


def create_instruction_pair(chunk: str) -> Dict[str, str]:
    """
    Create an instruction-response pair from a text chunk.

    Args:
        chunk: Text chunk to create an instruction pair from.

    Returns:
        Dict[str, str]: Dictionary with input_text (instruction) and output_text (response).
    """
    instruction = generate_instruction_for_chunk(chunk)
    
    return {
        "input_text": instruction,
        "output_text": chunk,
    }


def generate_training_examples(
    chunks: List[str],
    num_examples: int = 10,
    qa_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Generate training examples from text chunks.

    Args:
        chunks: List of text chunks.
        num_examples: Number of examples to generate.
        qa_ratio: Ratio of question-answer pairs to instruction-response pairs.

    Returns:
        List[Dict[str, str]]: List of training examples.
    """
    if not chunks:
        return []
    
    # Limit number of examples to number of chunks
    num_examples = min(num_examples, len(chunks))
    
    # Randomly select chunks
    selected_chunks = random.sample(chunks, num_examples)
    
    # Generate examples
    examples = []
    num_qa = int(num_examples * qa_ratio)
    
    for i, chunk in enumerate(selected_chunks):
        if i < num_qa:
            example = create_qa_pair(chunk)
        else:
            example = create_instruction_pair(chunk)
        
        examples.append(example)
    
    return examples


def validate_example(
    example: Dict[str, str],
    min_input_length: int = 10,
    max_input_length: int = 1024,
    min_output_length: int = 20,
    max_output_length: int = 2048,
) -> bool:
    """
    Validate a training example.

    Args:
        example: Training example to validate.
        min_input_length: Minimum length of input text.
        max_input_length: Maximum length of input text.
        min_output_length: Minimum length of output text.
        max_output_length: Maximum length of output text.

    Returns:
        bool: True if the example is valid, False otherwise.
    """
    # Check if example has required fields
    if "input_text" not in example or "output_text" not in example:
        return False
    
    # Check input length
    input_length = len(example["input_text"])
    if input_length < min_input_length or input_length > max_input_length:
        return False
    
    # Check output length
    output_length = len(example["output_text"])
    if output_length < min_output_length or output_length > max_output_length:
        return False
    
    return True


def split_data(
    examples: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split data into training, validation, and test sets.

    Args:
        examples: List of training examples.
        train_ratio: Ratio of examples to use for training.
        val_ratio: Ratio of examples to use for validation.
        test_ratio: Ratio of examples to use for testing.

    Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
            Training, validation, and test sets.
    """
    # Shuffle examples
    random.shuffle(examples)
    
    # Calculate split indices
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split data
    train_data = examples[:train_end]
    val_data = examples[train_end:val_end]
    test_data = examples[val_end:]
    
    return train_data, val_data, test_data


def save_jsonl(data: List[Dict], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save.
        file_path: Path to save the JSONL file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Saved {len(data)} examples to {file_path}")


def create_training_data(
    text_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[Dict] = None,
) -> Dict[str, int]:
    """
    Create training data from extracted text.

    Args:
        text_dir: Directory containing extracted text.
        output_dir: Directory to save training data.
        config: Configuration dictionary.

    Returns:
        Dict[str, int]: Dictionary with counts of examples in each split.
    """
    text_dir = Path(text_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default config if none provided
    if config is None:
        config = {
            "format": "jsonl",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "examples_per_document": 10,
            "min_input_length": 10,
            "max_input_length": 1024,
            "min_output_length": 20,
            "max_output_length": 2048,
        }
    
    # Load text chunks
    chunks_by_doc = load_text_chunks(text_dir)
    
    if not chunks_by_doc:
        logger.warning(f"No text chunks found in {text_dir}")
        return {"train": 0, "val": 0, "test": 0}
    
    # Generate examples for each document
    all_examples = []
    
    for doc_name, chunks in chunks_by_doc.items():
        examples = generate_training_examples(
            chunks,
            num_examples=config.get("examples_per_document", 10),
        )
        
        # Validate examples
        valid_examples = [
            example for example in examples
            if validate_example(
                example,
                min_input_length=config.get("min_input_length", 10),
                max_input_length=config.get("max_input_length", 1024),
                min_output_length=config.get("min_output_length", 20),
                max_output_length=config.get("max_output_length", 2048),
            )
        ]
        
        logger.info(f"Generated {len(valid_examples)} valid examples for {doc_name}")
        all_examples.extend(valid_examples)
    
    # Split data
    train_data, val_data, test_data = split_data(
        all_examples,
        train_ratio=config.get("train_split", 0.8),
        val_ratio=config.get("val_split", 0.1),
        test_ratio=config.get("test_split", 0.1),
    )
    
    # Save data
    if config.get("format", "jsonl").lower() == "jsonl":
        save_jsonl(train_data, output_dir / "train.jsonl")
        save_jsonl(val_data, output_dir / "val.jsonl")
        save_jsonl(test_data, output_dir / "test.jsonl")
    else:
        # Other formats could be implemented here
        logger.warning(f"Unsupported format: {config.get('format')}. Using JSONL.")
        save_jsonl(train_data, output_dir / "train.jsonl")
        save_jsonl(val_data, output_dir / "val.jsonl")
        save_jsonl(test_data, output_dir / "test.jsonl")
    
    return {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }
