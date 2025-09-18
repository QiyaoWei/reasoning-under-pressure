#!/usr/bin/env python3
"""
Monitor utilities: extract reasoning trace, create monitor prompt, extract monitor prediction, extract monitor reasoning.
"""

import re
from typing import Optional
from utils.prompts import (
    SYSTEM_PROMPT_REPORTER_DIAMONDS,
    SYSTEM_PROMPT_REPORTER_FUNCTION_CORRECTNESS,
    USER_PROMPT_REPORTER_DIAMONDS,
    USER_PROMPT_REPORTER_FUNCTION_CORRECTNESS
)


def extract_reasoning_trace(response: str) -> str:
    """Extract reasoning trace from response (everything before <measurements> tags)."""
    if "<measurements>" in response:
        # Split at the measurements tag and take everything before it
        reasoning = response.split("<measurements>")[0].strip()
        return reasoning
    # If no measurements tag, return the whole response
    return response.strip()


def create_monitor_prompt(reasoning_trace: str, dataset_name: str) -> str:
    """Create monitor prompt by concatenating system, user, and reasoning trace."""
    # Select system and user prompts based on dataset
    if dataset_name.startswith("diamonds-seed"):
        system_prompt = SYSTEM_PROMPT_REPORTER_DIAMONDS
        user_prompt = USER_PROMPT_REPORTER_DIAMONDS
    elif dataset_name == "function_correctness":
        system_prompt = SYSTEM_PROMPT_REPORTER_FUNCTION_CORRECTNESS
        user_prompt = USER_PROMPT_REPORTER_FUNCTION_CORRECTNESS
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return system_prompt + "\n" + user_prompt + "\n" + reasoning_trace


def extract_monitor_prediction(response_text: str) -> Optional[bool]:
    """Extract the boolean prediction from monitor's response."""
    # Look for content between <answer> tags
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip().lower()
        if 'true' in answer_text:
            return True
        elif 'false' in answer_text:
            return False
    return None


def extract_monitor_reasoning(response_text: str) -> str:
    """Extract the reasoning from monitor's response."""
    # Take everything before <answer> tags
    if "<answer>" in response_text:
        reasoning = response_text.split("<answer>")[0].strip()
        return reasoning
    else:
        return response_text.strip()
