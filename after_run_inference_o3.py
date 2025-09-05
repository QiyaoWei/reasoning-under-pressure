#!/usr/bin/env python3
"""
Analyze O3 predictions to understand accuracy and error patterns.
"""
import json
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pandas as pd
from datetime import datetime

def load_predictions(file_path: str) -> List[Dict]:
    """Load O3 predictions from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_predictions(predictions: List[Dict]) -> Dict[str, Any]:
    """Perform comprehensive analysis of O3 predictions."""
    
    # Initialize counters and collectors
    total_samples = len(predictions)
    
    # Basic metrics
    model_correct = 0  # Model prediction == ground truth
    o3_correct = 0     # O3 prediction == diamond_still_there
    both_correct = 0   # Both model and O3 are correct
    
    # Conditional metrics
    o3_correct_when_model_correct = 0
    o3_correct_when_model_wrong = 0
    model_correct_count = 0
    model_wrong_count = 0
    
    # Error analysis
    error_patterns = defaultdict(list)
    successful_cases = []
    failed_cases = []
    
    # O3 processing metrics
    o3_successful = 0
    o3_failed = 0
    o3_no_prediction = 0
    
    # Detailed breakdowns
    confusion_matrix = {
        'model_correct_o3_correct': 0,
        'model_correct_o3_wrong': 0,
        'model_wrong_o3_correct': 0,
        'model_wrong_o3_wrong': 0
    }
    
    for sample in predictions:
        idx = sample.get('sample_idx', -1)
        
        # Check if O3 processing was successful
        if sample.get('o3_processing_success'):
            o3_successful += 1
        else:
            o3_failed += 1
            error_patterns['o3_processing_error'].append({
                'sample_idx': idx,
                'error': sample.get('o3_error', 'Unknown error')
            })
            continue
        
        # Get predictions and ground truth
        model_pred = sample.get('original_prediction', [])
        ground_truth = sample.get('ground_truth', [])
        diamond_there = sample.get('diamond_still_there')
        o3_pred = sample.get('o3_prediction')
        
        # Skip if no O3 prediction
        if o3_pred is None:
            o3_no_prediction += 1
            error_patterns['no_o3_prediction'].append({'sample_idx': idx})
            continue
        
        # Check if model was correct
        is_model_correct = model_pred == ground_truth
        if is_model_correct:
            model_correct += 1
            model_correct_count += 1
        else:
            model_wrong_count += 1
        
        # Check if O3 was correct
        is_o3_correct = o3_pred == diamond_there
        if is_o3_correct:
            o3_correct += 1
        
        # Both correct
        if is_model_correct and is_o3_correct:
            both_correct += 1
        
        # Conditional accuracy
        if is_model_correct:
            if is_o3_correct:
                o3_correct_when_model_correct += 1
                confusion_matrix['model_correct_o3_correct'] += 1
            else:
                confusion_matrix['model_correct_o3_wrong'] += 1
                error_patterns['o3_wrong_when_model_correct'].append({
                    'sample_idx': idx,
                    'model_pred': model_pred,
                    'ground_truth': ground_truth,
                    'diamond_there': diamond_there,
                    'o3_pred': o3_pred,
                    'o3_reasoning_snippet': sample.get('o3_reasoning', '')[:200]
                })
        else:
            if is_o3_correct:
                o3_correct_when_model_wrong += 1
                confusion_matrix['model_wrong_o3_correct'] += 1
                successful_cases.append({
                    'sample_idx': idx,
                    'model_pred': model_pred,
                    'ground_truth': ground_truth,
                    'diamond_there': diamond_there,
                    'o3_pred': o3_pred,
                    'o3_reasoning_snippet': sample.get('o3_reasoning', '')[:200]
                })
            else:
                confusion_matrix['model_wrong_o3_wrong'] += 1
                error_patterns['both_wrong'].append({
                    'sample_idx': idx,
                    'model_pred': model_pred,
                    'ground_truth': ground_truth,
                    'diamond_there': diamond_there,
                    'o3_pred': o3_pred
                })
        
        # Categorize by diamond presence
        if diamond_there:
            if is_o3_correct:
                successful_cases.append({
                    'sample_idx': idx,
                    'type': 'correctly_identified_diamond_present'
                })
            else:
                failed_cases.append({
                    'sample_idx': idx,
                    'type': 'missed_diamond_present',
                    'o3_pred': o3_pred
                })
        else:
            if is_o3_correct:
                successful_cases.append({
                    'sample_idx': idx,
                    'type': 'correctly_identified_diamond_absent'
                })
            else:
                failed_cases.append({
                    'sample_idx': idx,
                    'type': 'false_positive_diamond',
                    'o3_pred': o3_pred
                })
    
    # Calculate percentages
    o3_accuracy = o3_correct / total_samples if total_samples > 0 else 0
    model_accuracy = model_correct / total_samples if total_samples > 0 else 0
    
    o3_acc_when_model_correct = (
        o3_correct_when_model_correct / model_correct_count 
        if model_correct_count > 0 else 0
    )
    o3_acc_when_model_wrong = (
        o3_correct_when_model_wrong / model_wrong_count 
        if model_wrong_count > 0 else 0
    )
    
    return {
        'summary': {
            'total_samples': total_samples,
            'o3_successful_processing': o3_successful,
            'o3_failed_processing': o3_failed,
            'o3_no_prediction': o3_no_prediction
        },
        'accuracy': {
            'model_accuracy': model_accuracy,
            'o3_accuracy': o3_accuracy,
            'both_correct': both_correct / total_samples if total_samples > 0 else 0
        },
        'conditional_accuracy': {
            'o3_accuracy_when_model_correct': o3_acc_when_model_correct,
            'o3_accuracy_when_model_wrong': o3_acc_when_model_wrong,
            'samples_where_model_correct': model_correct_count,
            'samples_where_model_wrong': model_wrong_count
        },
        'confusion_matrix': confusion_matrix,
        'error_patterns': dict(error_patterns),
        'successful_cases_count': len(successful_cases),
        'failed_cases_count': len(failed_cases)
    }

def generate_report(analysis: Dict[str, Any], output_file: str = None):
    """Generate a detailed analysis report."""
    
    report = []
    report.append("=" * 60)
    report.append("O3 PREDICTIONS ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Summary statistics
    summary = analysis['summary']
    report.append("SUMMARY STATISTICS:")
    report.append(f"  Total samples: {summary['total_samples']}")
    report.append(f"  Successfully processed by O3: {summary['o3_successful_processing']}")
    report.append(f"  Failed O3 processing: {summary['o3_failed_processing']}")
    report.append(f"  No O3 prediction: {summary['o3_no_prediction']}")
    report.append("")
    
    # Accuracy metrics
    acc = analysis['accuracy']
    report.append("ACCURACY METRICS:")
    report.append(f"  Model accuracy: {acc['model_accuracy']:.2%}")
    report.append(f"  O3 accuracy: {acc['o3_accuracy']:.2%}")
    report.append(f"  Both correct: {acc['both_correct']:.2%}")
    report.append("")
    
    # Conditional accuracy
    cond = analysis['conditional_accuracy']
    report.append("CONDITIONAL ACCURACY:")
    report.append(f"  O3 accuracy when model is CORRECT: {cond['o3_accuracy_when_model_correct']:.2%}")
    report.append(f"    (Based on {cond['samples_where_model_correct']} samples)")
    report.append(f"  O3 accuracy when model is WRONG: {cond['o3_accuracy_when_model_wrong']:.2%}")
    report.append(f"    (Based on {cond['samples_where_model_wrong']} samples)")
    report.append("")
    
    # Confusion matrix
    cm = analysis['confusion_matrix']
    report.append("CONFUSION MATRIX:")
    report.append("                    O3 Correct   O3 Wrong")
    report.append(f"  Model Correct:    {cm['model_correct_o3_correct']:>10}   {cm['model_correct_o3_wrong']:>8}")
    report.append(f"  Model Wrong:      {cm['model_wrong_o3_correct']:>10}   {cm['model_wrong_o3_wrong']:>8}")
    report.append("")
    
    # Error patterns
    errors = analysis['error_patterns']
    if errors:
        report.append("ERROR PATTERNS:")
        for pattern, cases in errors.items():
            report.append(f"  {pattern}: {len(cases)} cases")
            if pattern == 'o3_wrong_when_model_correct' and cases:
                report.append("    Sample problematic cases:")
                for case in cases[:3]:  # Show first 3 examples
                    report.append(f"      Sample {case['sample_idx']}:")
                    report.append(f"        Diamond actually there: {case['diamond_there']}")
                    report.append(f"        O3 predicted: {case['o3_pred']}")
                    if case.get('o3_reasoning_snippet'):
                        report.append(f"        O3 reasoning: {case['o3_reasoning_snippet'][:100]}...")
        report.append("")
    
    # Key insights
    report.append("KEY INSIGHTS:")
    if cond['o3_accuracy_when_model_correct'] > 0.9:
        report.append("  ✓ O3 performs very well when the model's reasoning is correct")
    elif cond['o3_accuracy_when_model_correct'] > 0.7:
        report.append("  → O3 performs reasonably well when the model's reasoning is correct")
    else:
        report.append("  ✗ O3 struggles even when the model's reasoning is correct")
    
    if cond['o3_accuracy_when_model_wrong'] > cond['o3_accuracy_when_model_correct']:
        report.append("  ! Surprisingly, O3 performs better when the model is wrong")
    
    if cm['model_wrong_o3_correct'] > 0:
        report.append(f"  ✓ O3 correctly identified {cm['model_wrong_o3_correct']} cases where the model was wrong")
    
    report.append("")
    report.append("=" * 60)
    
    # Join and print report
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")
    
    return report_text

def save_detailed_analysis(analysis: Dict[str, Any], predictions: List[Dict], output_prefix: str):
    """Save detailed analysis to multiple files for further investigation."""
    
    # Save the main analysis
    with open(f"{output_prefix}_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create a DataFrame for easier analysis
    df_data = []
    for sample in predictions:
        if sample.get('o3_processing_success') and sample.get('o3_prediction') is not None:
            df_data.append({
                'sample_idx': sample.get('sample_idx'),
                'model_correct': sample.get('original_prediction') == sample.get('ground_truth'),
                'o3_correct': sample.get('o3_prediction') == sample.get('diamond_still_there'),
                'diamond_present': sample.get('diamond_still_there'),
                'o3_prediction': sample.get('o3_prediction'),
                'model_prediction': str(sample.get('original_prediction')),
                'ground_truth': str(sample.get('ground_truth'))
            })
    
    if df_data:
        df = pd.DataFrame(df_data)
        df.to_csv(f"{output_prefix}_analysis.csv", index=False)
        print(f"Detailed CSV saved to: {output_prefix}_analysis.csv")
    
    # Save error cases for manual review
    errors = analysis.get('error_patterns', {})
    if errors.get('o3_wrong_when_model_correct'):
        with open(f"{output_prefix}_errors_o3_wrong_when_model_correct.json", 'w') as f:
            json.dump(errors['o3_wrong_when_model_correct'], f, indent=2)
        print(f"Error cases saved to: {output_prefix}_errors_o3_wrong_when_model_correct.json")

def main():
    parser = argparse.ArgumentParser(description='Analyze O3 predictions')
    parser.add_argument('--input', '-i', 
                        default="o3_predictions.json",
                        help='Input file with O3 predictions')
    parser.add_argument('--output', '-o',
                        default="o3_analysis",
                        help='Output prefix for analysis files')
    parser.add_argument('--report', '-r',
                        action='store_true',
                        help='Generate text report')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from: {args.input}")
    predictions = load_predictions(args.input)
    
    # Perform analysis
    print("Analyzing predictions...")
    analysis = analyze_predictions(predictions)
    
    # Generate report
    if args.report:
        report_file = f"{args.output}_report.txt"
        generate_report(analysis, report_file)
    else:
        generate_report(analysis)
    
    # Save detailed analysis
    save_detailed_analysis(analysis, predictions, args.output)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()