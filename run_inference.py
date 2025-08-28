#!/usr/bin/env python3
"""
Inference script for VERL-trained models.
Supports loading from FSDP checkpoints or HuggingFace format.
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from VERL checkpoint directory."""
    
    # Check if HuggingFace format exists
    hf_path = os.path.join(checkpoint_path, "actor", "huggingface")
    if os.path.exists(hf_path):
        print(f"Loading model from HuggingFace format at: {hf_path}")
        model = AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    else:
        # Try loading from actor directory directly
        actor_path = os.path.join(checkpoint_path, "actor")
        if os.path.exists(actor_path):
            print(f"Loading model from actor directory: {actor_path}")
            model = AutoModelForCausalLM.from_pretrained(
                actor_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(actor_path, trust_remote_code=True)
        else:
            raise ValueError(f"Could not find valid model checkpoint at {checkpoint_path}")
    
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> List[str]:
    """Generate responses for a list of prompts."""
    
    responses = []
    for prompt in prompts:
        # Format as chat if needed
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(response)
        
    return responses


def main():
    parser = argparse.ArgumentParser(description="Run inference with VERL-trained models")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/verl_grpo_diamonds/qwen2.5_coder_7b_diamonds/global_step_40)"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="List of prompts to generate responses for"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Get prompts
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    
    if not prompts:
        # Default example prompts
        prompts = [
            "Write a Python function to calculate the factorial of a number.",
            "Explain what recursion is in programming.",
            "How do I reverse a string in Python?"
        ]
        print("No prompts provided. Using default examples:")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p[:50]}...")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate responses
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    
    # Display results
    results = []
    for prompt, response in zip(prompts, responses):
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"RESPONSE: {response}")
        results.append({"prompt": prompt, "response": response})
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()