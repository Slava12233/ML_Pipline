#!/usr/bin/env python3
"""
Test Model CLI - A command-line tool for testing fine-tuned models.

This script provides an interactive interface to test models both locally
and on Vertex AI. It supports direct querying, PDF-based testing, and
comparison between standard models and PEFT-adapted models.
"""

import os
import sys
import argparse
import logging
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utilities
from src.utils.config import get_config
from src.evaluation.model_evaluator import evaluate_model_on_pdf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_local_model(
    model_dir: str,
    questions: List[str] = None,
    interactive: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 200,
):
    """Test a locally available model with direct prompts."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        logger.info(f"Loading model from {model_dir}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Interactive mode
        if interactive:
            print("\n" + "=" * 50)
            print("Interactive Testing Session")
            print("Type 'quit' to exit")
            print("=" * 50 + "\n")
            
            while True:
                # Get user input
                user_input = input("\nEnter your question: ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Prepare input
                prompt = f"Question: {user_input}\nAnswer:"
                
                # Tokenize input
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                
                # Generate response
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=temperature
                    )
                
                # Decode output
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract just the answer part
                answer_parts = response.split("Answer:")
                answer = answer_parts[-1].strip() if len(answer_parts) > 1 else response.strip()
                
                # Print response
                print("\nResponse:")
                print("-" * 50)
                print(answer)
                print("-" * 50)
        
        # Pre-defined questions mode
        elif questions:
            results = []
            
            for i, question in enumerate(questions):
                print(f"\nQuestion {i+1}: {question}")
                print("-" * 50)
                
                # Prepare input
                prompt = f"Question: {question}\nAnswer:"
                
                # Tokenize input
                input_ids = tokenizer.encode(prompt, return_tensors="pt")
                
                # Generate response
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=temperature
                    )
                
                # Decode output
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract just the answer part
                answer_parts = response.split("Answer:")
                answer = answer_parts[-1].strip() if len(answer_parts) > 1 else response.strip()
                
                # Print response
                print("Response:")
                print(answer)
                print("-" * 50)
                
                # Save result
                results.append({
                    "question": question,
                    "answer": answer
                })
            
            # Save results to file
            output_file = Path(model_dir) / "test_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Test results saved to {output_file}")
        
        else:
            logger.error("Either interactive mode or a list of questions must be provided")
    
    except Exception as e:
        logger.error(f"Error testing local model: {str(e)}")
        raise


def test_vertex_model(
    endpoint_id: str,
    project_id: str,
    location: str,
    questions: List[str],
    temperature: float = 0.7,
    max_tokens: int = 200,
    output_file: Optional[str] = None,
):
    """Test a model deployed on Vertex AI Endpoints."""
    try:
        from google.cloud import aiplatform
        
        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location=location,
        )
        
        # Get endpoint
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        results = []
        
        for i, question in enumerate(questions):
            print(f"\nQuestion {i+1}: {question}")
            print("-" * 50)
            
            # Create request
            instances = [
                {
                    "prompt": f"Question: {question}\nAnswer:",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            ]
            
            # Query endpoint
            response = endpoint.predict(instances=instances)
            
            # Extract answer
            prediction = response.predictions[0]
            answer = prediction.get("content", prediction)
            
            # Print response
            print("Response:")
            print(answer)
            print("-" * 50)
            
            # Save result
            results.append({
                "question": question,
                "answer": answer
            })
        
        # Save results to file
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Test results saved to {output_file}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error testing Vertex AI model: {str(e)}")
        raise


def test_model_on_pdf(
    pdf_path: str,
    model_dir: str = None,
    endpoint_id: str = None,
    project_id: str = None,
    location: str = None,
    use_standard: bool = True,
    use_peft: bool = True,
    num_questions: int = 3,
    output_path: Optional[str] = None,
):
    """Test model(s) on content from a PDF file."""
    try:
        # Local testing mode
        if model_dir:
            evaluate_model_on_pdf(
                pdf_path=pdf_path,
                model_name="gpt2",  # Base model
                use_standard=use_standard,
                use_peft=use_peft,
                num_questions=num_questions,
                output_path=output_path
            )
        
        # Vertex AI testing mode
        elif endpoint_id and project_id and location:
            # Extract text from PDF
            from src.pdf_processing.extract import extract_text_with_pypdf2
            from src.data_preparation.generate import generate_questions_from_chunk
            
            # Extract text from PDF
            text = extract_text_with_pypdf2(pdf_path)
            
            # Generate questions
            questions = generate_questions_from_chunk(text, num_questions=num_questions)
            
            # Test on Vertex AI
            results = test_vertex_model(
                endpoint_id=endpoint_id,
                project_id=project_id,
                location=location,
                questions=questions,
                output_file=output_path
            )
            
            return results
        
        else:
            logger.error("Either model_dir or endpoint_id/project_id/location must be provided")
    
    except Exception as e:
        logger.error(f"Error testing model on PDF: {str(e)}")
        raise


def main():
    """Run the model testing CLI."""
    parser = argparse.ArgumentParser(description="Test fine-tuned models")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--local", action="store_true", help="Test local model")
    mode_group.add_argument("--vertex", action="store_true", help="Test model on Vertex AI")
    
    # Common arguments
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--env", type=str, choices=["local", "vertex", "production"], default="local", help="Environment configuration to use")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--questions", type=str, nargs="+", help="List of questions to test")
    parser.add_argument("--questions-file", type=str, help="Path to JSON file containing questions")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to test on")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens to generate")
    
    # Local testing arguments
    parser.add_argument("--model-dir", type=str, default="data/model", help="Path to model directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    parser.add_argument("--use-standard", action="store_true", help="Use standard model")
    parser.add_argument("--use-peft", action="store_true", help="Use PEFT-adapted model")
    
    # Vertex AI arguments
    parser.add_argument("--endpoint-id", type=str, help="Vertex AI Endpoint ID")
    parser.add_argument("--project-id", type=str, help="GCP Project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP Region")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(args.config, env=args.env)
    
    # Get questions from file if provided
    questions = args.questions or []
    if args.questions_file:
        try:
            with open(args.questions_file, "r") as f:
                questions_data = json.load(f)
                if isinstance(questions_data, list):
                    questions.extend(questions_data)
                elif isinstance(questions_data, dict) and "questions" in questions_data:
                    questions.extend(questions_data["questions"])
        except Exception as e:
            logger.error(f"Error loading questions from file: {str(e)}")
    
    # Local testing mode
    if args.local:
        if args.pdf:
            logger.info(f"Testing local model on PDF: {args.pdf}")
            
            # Create output path if not provided
            output_path = args.output
            if not output_path:
                pdf_name = Path(args.pdf).stem
                output_path = Path(args.model_dir) / f"evaluation_{pdf_name}.yaml"
            
            test_model_on_pdf(
                pdf_path=args.pdf,
                model_dir=args.model_dir,
                use_standard=args.use_standard or True,  # Default to True if not specified
                use_peft=args.use_peft or True,  # Default to True if not specified
                num_questions=3,
                output_path=output_path
            )
        else:
            logger.info(f"Testing local model from {args.model_dir}")
            test_local_model(
                model_dir=args.model_dir,
                questions=questions,
                interactive=args.interactive,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
    
    # Vertex AI testing mode
    elif args.vertex:
        # Get Vertex AI configuration
        endpoint_id = args.endpoint_id or config.deployment.endpoint_id
        project_id = args.project_id or config.gcp.project_id
        location = args.location or config.gcp.region
        
        if not endpoint_id:
            logger.error("Endpoint ID must be provided for Vertex AI testing")
            return
        
        if not project_id:
            logger.error("Project ID must be provided for Vertex AI testing")
            return
        
        if args.pdf:
            logger.info(f"Testing Vertex AI model on PDF: {args.pdf}")
            
            # Create output path if not provided
            output_path = args.output
            if not output_path:
                pdf_name = Path(args.pdf).stem
                output_path = f"evaluation_{pdf_name}_vertex.json"
            
            test_model_on_pdf(
                pdf_path=args.pdf,
                endpoint_id=endpoint_id,
                project_id=project_id,
                location=location,
                num_questions=3,
                output_path=output_path
            )
        else:
            if not questions:
                logger.error("Questions must be provided for Vertex AI testing")
                return
            
            logger.info(f"Testing Vertex AI model on endpoint: {endpoint_id}")
            test_vertex_model(
                endpoint_id=endpoint_id,
                project_id=project_id,
                location=location,
                questions=questions,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_file=args.output
            )


if __name__ == "__main__":
    main() 