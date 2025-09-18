"""
Utility functions for extracting and processing measurements from model outputs.
"""

import os
import re  

def extract_measurements(text: str, dataset_name: str = "diamonds-seed0") -> list:
    """Extracts the measurements list from model output.
    
    Args:
        text: The model output text to parse
        dataset_name: Name of the dataset to validate against.
    
    Returns:
        List of boolean measurements or None if parsing fails
    """
    try:
        match = re.search(r'<measurements>\s*\[(.*?)\]\s*</measurements>', text, re.DOTALL)
        if match:
            list_content = match.group(1).strip()
            measurements = []
            for item in list_content.split(','):
                item = item.strip().lower()
                if item == 'true':
                    measurements.append(True)
                elif item == 'false':
                    measurements.append(False)
                else:
                    print(f"Invalid measurement value: {item}")
                    return None
            
            if (len(measurements) == 3) and dataset_name.startswith("diamonds-seed"):
                # diamonds dataset - correct number of measurements
                return measurements
            elif dataset_name == "function_correctness":
                # check number of measurements later
                return measurements
            else:
                # wrong number of measurements for diamonds dataset
                print(f"Wrong number of measurements: {len(measurements)}")
                return None
        else:
            # print(f"Failed to extract measurements from output: {text}")
            return None
    except Exception as e:
        print(f"Error extracting measurements: {e}")
        return None 