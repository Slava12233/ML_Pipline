# DATA_PREPARATION_TASK.md - PDF Fine-tuning Data Preparation Plan

## Overview

This document outlines the specific steps for enhancing our data preparation pipeline to create high-quality training data from PDF documents. Our focus is on extracting text accurately, generating relevant question-answer pairs, and ensuring the data is properly formatted for fine-tuning.

## Target Timeline
- **PDF Processing Improvements**: 2 days
- **Training Data Generation Enhancements**: 3 days
- **Data Quality Metrics**: 1 day
- **GCS Integration**: 1 day

## Prerequisites

- PDF files in `data/pdfs` directory
- Python environment with required packages installed
- GCP credentials configured (for GCS upload)

## Data Preparation Tasks

### 1. Enhance PDF Text Extraction

#### Task 1.1: Improve Text Extraction Accuracy
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/pdf_processing/extract.py`
- **Function:** `extract_text_with_pypdf2()` and `extract_text_with_document_ai()`
- **Input:** PDF files
- **Output:** Extracted text with improved formatting

Specific improvements needed:
- Fix newline handling to preserve paragraph structure
- Improve table extraction
- Fix encoding issues with special characters

```python
# Example extraction call
python -m src.main process-pdfs data/pdfs data/extracted_text --config-path config/modified_config.yaml
```

#### Task 1.2: Implement Document AI Integration
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/pdf_processing/extract.py`
- **Function:** `extract_text_with_document_ai()`
- **Task:** Complete the Document AI integration for complex PDF processing

```python
def extract_text_with_document_ai(pdf_path, config):
    """Extract text using Google Document AI."""
    from google.cloud import documentai_v1 as documentai
    
    # Get processor ID from config
    processor_id = config.get("document_ai", {}).get("processor_id")
    
    # Initialize Document AI client
    client = documentai.DocumentProcessorServiceClient()
    
    # Read PDF file
    with open(pdf_path, "rb") as f:
        content = f.read()
    
    # Process document
    # [Implementation details here]
    
    return extracted_text
```

### 2. Improve Training Data Generation

#### Task 2.1: Enhance Question Generation
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/data_preparation/generate.py`
- **Function:** `generate_questions_from_chunk()`
- **Task:** Generate more diverse and contextually relevant questions

Implement the following improvements:
- Use different question patterns (not just "What can you tell me about...")
- Generate questions based on key entities identified in the text
- Ensure questions are relevant to the content

```python
def generate_questions_from_chunk(chunk_text, num_questions=3):
    """Generate questions from a text chunk."""
    # Extract key entities or concepts from the text
    entities = extract_entities(chunk_text)
    
    questions = []
    
    # Generate questions using different patterns
    question_patterns = [
        "What can you tell me about {entity}?",
        "How does {entity} work?",
        "Why is {entity} important?",
        "What are the key features of {entity}?",
        "Explain the concept of {entity}."
    ]
    
    # [Implementation details here]
    
    return questions
```

#### Task 2.2: Improve Answer Quality
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/data_preparation/generate.py`
- **Function:** `generate_answer_for_question()`
- **Task:** Generate more comprehensive and accurate answers

### 3. Data Quality Improvements

#### Task 3.1: Add Data Validation
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/data_preparation/validate.py`
- **Function:** `validate_training_data()`
- **Task:** Implement comprehensive validation of training data

Validation rules to implement:
- Ensure minimum/maximum lengths for inputs and outputs
- Check for duplicates
- Verify question-answer relevance
- Detect and fix formatting issues

```python
def validate_training_data(data_dir):
    """Validate training data files."""
    issues = []
    
    # Load training data
    train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(data_dir, "val.jsonl"))
    test_data = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    
    # Check for duplicates
    all_examples = train_data + val_data + test_data
    input_texts = [ex["input_text"] for ex in all_examples]
    duplicate_indices = find_duplicates(input_texts)
    
    if duplicate_indices:
        issues.append(f"Found {len(duplicate_indices)} duplicate examples")
    
    # Check length constraints
    # [More validation checks]
    
    return issues
```

#### Task 3.2: Implement Data Augmentation
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/data_preparation/augment.py`
- **Task:** Create methods to augment training data for better model performance

Implement techniques such as:
- Paraphrasing questions
- Generating variations of answers
- Creating examples with different formatting

### 4. GCS Integration Improvements

#### Task 4.1: Enhance GCS Export
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/data_preparation/export.py`
- **Function:** `export_training_data_to_gcs()`
- **Task:** Improve GCS export functionality with better error handling and progress reporting

```python
def export_training_data_to_gcs(
    data_dir, 
    project_id, 
    bucket_name, 
    gcs_dir
):
    """Export training data to Google Cloud Storage."""
    from google.cloud import storage
    
    # Initialize storage client
    storage_client = storage.Client(project=project_id)
    
    # Get bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Export data files
    gcs_uris = {}
    
    for split in ["train", "val", "test"]:
        source_path = os.path.join(data_dir, f"{split}.jsonl")
        destination_blob_name = f"{gcs_dir}/{split}.jsonl"
        
        if os.path.exists(source_path):
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_path)
            gcs_uris[split] = f"gs://{bucket_name}/{destination_blob_name}"
    
    return gcs_uris
```

#### Task 4.2: Add Data Versioning
- **Developer:** [ASSIGN DEVELOPER]
- **File Path:** `src/data_preparation/versioning.py`
- **Task:** Implement simple versioning for training data to track changes

## Technical Implementation Details

### Text Extraction Logic

The PDF processing module should handle various PDF formats and structures. Focus on these key areas:

1. **Layout Analysis**: Preserve the document structure when extracting text
2. **Table Handling**: Extract tabular data in a structured format
3. **Special Characters**: Handle special characters and symbols correctly
4. **Metadata Extraction**: Extract and use document metadata for context

Example text extraction workflow:

```python
def process_pdf(pdf_path, output_dir, method="pypdf2", config=None):
    """Process a PDF file and extract text."""
    # Choose extraction method
    if method == "document_ai":
        text = extract_text_with_document_ai(pdf_path, config)
    else:
        text = extract_text_with_pypdf2(pdf_path)
    
    # Post-process text
    text = clean_text(text)
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size=config.get("chunk_size", 1000))
    
    # Save results
    save_results(text, chunks, output_dir, pdf_path.stem)
    
    return {
        "text_file": f"{output_dir}/{pdf_path.stem}.txt",
        "chunks_dir": f"{output_dir}/{pdf_path.stem}_chunks"
    }
```

### Training Data Generation

The training data generation process should be focused on quality over quantity. Some key considerations:

1. **Context Preservation**: Ensure each example has enough context
2. **Question Diversity**: Generate diverse question types
3. **Answer Accuracy**: Answers should be factually correct and relevant
4. **Format Consistency**: Maintain consistent format for fine-tuning

Example question generation workflow:

```python
def create_training_data(text_dir, output_dir, config):
    """Create training data from extracted text."""
    all_examples = []
    
    # Process each text file
    for text_file in Path(text_dir).glob("*.txt"):
        # Load text
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Load chunks
        chunks_dir = Path(text_dir) / f"{text_file.stem}_chunks"
        chunks = load_chunks(chunks_dir)
        
        # Generate examples from chunks
        examples = []
        for chunk in chunks:
            # Generate questions for the chunk
            questions = generate_questions_from_chunk(chunk, num_questions=3)
            
            # Generate answers for each question
            for question in questions:
                answer = generate_answer_for_question(question, chunk)
                
                # Create example
                example = {
                    "input_text": question,
                    "output_text": answer
                }
                
                examples.append(example)
        
        all_examples.extend(examples)
    
    # Split into train/val/test
    # [Split implementation]
    
    return {
        "train": len(train_examples),
        "val": len(val_examples),
        "test": len(test_examples)
    }
```

## Data Quality Metrics

We need to implement the following quality metrics for our training data:

- **Coverage**: Percentage of the source document represented in training data
- **Diversity**: Variety of question types and formats
- **Relevance**: How well answers match questions
- **Correctness**: Factual accuracy of answers

## Success Criteria

1. Improved text extraction from complex PDFs
2. More diverse and relevant question-answer pairs
3. Higher quality metrics for training data
4. Successful export to GCS with versioning

## Notes for Development Team

- Focus on quality over quantity for training examples
- Test with a variety of PDF types (text-heavy, tables, images, etc.)
- Document any new configuration parameters
- Write unit tests for all new functionality
- Consider the tradeoff between automated generation and quality 