"""
Test to verify BatchRewardManager interface works correctly.
This tests the modified compute_score function with batch calls.

Run with:
    pytest our_tests/test_batch_interface.py -v -s
"""

import pytest
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_batch_interface_with_monitor():
    """Test that batch interface works correctly with monitor enabled"""
    
    # Set environment before importing
    os.environ["TRAIN_WITH_MONITOR"] = "true"
    from rewards import compute_score
    
    print("\n🧪 Testing BatchRewardManager interface...")
    
    # Load a small subset of responses for testing
    responses_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_responses_expanded.txt")
    with open(responses_file, 'r') as f:
        all_responses = [line.strip() for line in f.readlines() if line.strip()]
    
    # Use first 10 responses for quick test
    test_responses = all_responses[:10]
    test_ground_truths = []
    test_extra_infos = []
    test_data_sources = []
    
    for response in test_responses:
        if "[true]" in response:
            test_ground_truths.append("[true]")
            test_extra_infos.append({"is_correct": True})
        else:
            test_ground_truths.append("[false]")
            test_extra_infos.append({"is_correct": False})
        test_data_sources.append("function_correctness")
    
    print(f"Testing with {len(test_responses)} samples...")
    
    # Test batch call (simulating BatchRewardManager)
    start_time = time.time()
    
    batch_scores = compute_score(
        data_sources=test_data_sources,
        solution_strs=test_responses,
        ground_truths=test_ground_truths,
        extra_infos=test_extra_infos
    )
    
    batch_time = time.time() - start_time
    
    print(f"✅ Batch call completed in {batch_time:.3f} seconds")
    print(f"   Returned {len(batch_scores)} scores")
    print(f"   Sample scores: {[result['score'] for result in batch_scores[:3]]}")
    
    # Verify results
    assert isinstance(batch_scores, list), "Batch call should return list of results"
    assert len(batch_scores) == len(test_responses), "Should return one result per response"
    assert all(isinstance(result, dict) for result in batch_scores), "All results should be dictionaries"
    assert all("score" in result for result in batch_scores), "All results should have score field"
    
    print("✅ Batch interface test passed!")


def test_single_interface_compatibility():
    """Test that single interface still works (backward compatibility)"""
    
    # Set environment before importing
    os.environ["TRAIN_WITH_MONITOR"] = "true"
    from rewards import compute_score
    
    print("\n🔄 Testing backward compatibility...")
    
    # Test single call (traditional interface)
    response = "Looking at this code, I believe it works correctly. <measurements>[true]</measurements>"
    ground_truth = "[true]"
    extra_info = {"is_correct": True}
    
    start_time = time.time()
    
    single_result = compute_score(
        response=response,
        ground_truth=ground_truth,
        dataset_name="function_correctness",
        extra_info=extra_info
    )
    
    single_time = time.time() - start_time
    
    print(f"✅ Single call completed in {single_time:.3f} seconds")
    print(f"   Returned result: {single_result}")
    
    # Verify results
    assert isinstance(single_result, dict), "Single call should return dict result"
    assert "score" in single_result, "Result should contain score field"
    assert isinstance(single_result["score"], (int, float)), "Score should be numeric"
    
    print("✅ Backward compatibility test passed!")


def test_performance_difference():
    """Quick test to see if batch interface is faster than multiple single calls"""
    
    # Set environment before importing
    os.environ["TRAIN_WITH_MONITOR"] = "true"
    from rewards import compute_score
    
    print("\n⚡ Testing performance difference...")
    
    # Use small sample for quick comparison
    responses_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_responses_expanded.txt")
    with open(responses_file, 'r') as f:
        all_responses = [line.strip() for line in f.readlines() if line.strip()]
    
    # Use first 5 responses for quick test
    test_responses = all_responses[:5]
    test_ground_truths = []
    test_extra_infos = []
    
    for response in test_responses:
        if "[true]" in response:
            test_ground_truths.append("[true]")
            test_extra_infos.append({"is_correct": True})
        else:
            test_ground_truths.append("[false]")
            test_extra_infos.append({"is_correct": False})
    
    print(f"Comparing approaches with {len(test_responses)} samples...")
    
    # Method 1: Multiple single calls (current naive approach)
    start_time = time.time()
    single_scores = []
    for response, ground_truth, extra_info in zip(test_responses, test_ground_truths, test_extra_infos):
        result = compute_score(
            response=response,
            ground_truth=ground_truth,
            dataset_name="function_correctness",
            extra_info=extra_info
        )
        single_scores.append(result["score"])
    single_time = time.time() - start_time
    
    # Method 2: Single batch call (BatchRewardManager approach)
    start_time = time.time()
    batch_scores = compute_score(
        data_sources=["function_correctness"] * len(test_responses),
        solution_strs=test_responses,
        ground_truths=test_ground_truths,
        extra_infos=test_extra_infos
    )
    batch_time = time.time() - start_time
    
    # Results
    print(f"📊 Performance Comparison:")
    print(f"   Multiple single calls: {single_time:.3f} seconds")
    print(f"   Single batch call:     {batch_time:.3f} seconds")
    print(f"   Speedup factor:        {single_time/batch_time:.2f}x")
    
    # Verify both methods give same results
    assert len(single_scores) == len(batch_scores), "Both methods should return same number of scores"
    
    # Compare the actual scores (extract from dicts for batch results)
    single_score_values = [score["score"] if isinstance(score, dict) else score for score in single_scores]
    batch_score_values = [result["score"] for result in batch_scores]
    
    for i, (single_val, batch_val) in enumerate(zip(single_score_values, batch_score_values)):
        assert abs(single_val - batch_val) < 1e-6, f"Score mismatch at index {i}: {single_val} vs {batch_val}"
    
    print(f"   Results match: ✅")
    
    print("✅ Performance comparison completed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
