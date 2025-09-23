"""
Test that non-monitor rewards (format, correctness, verbosity) work correctly
with the BatchRewardManager interface, and that monitor can be disabled.

This ensures our batch optimization doesn't break the core reward components.

Run with:
    pytest our_tests/integration/test_non_monitor_rewards.py -v -s
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_non_monitor_rewards_single_call():
    """
    Test that format, correctness, and verbosity rewards work in single-call mode
    with monitor disabled.
    """
    print("\n" + "="*80)
    print("NON-MONITOR REWARDS TEST: SINGLE CALL")
    print("="*80)
    
    # Disable monitor for this test
    original_monitor = os.environ.get("TRAIN_WITH_MONITOR")
    os.environ["TRAIN_WITH_MONITOR"] = "false"
    
    try:
        # Import AFTER setting environment
        from rewards import compute_score
        
        # Test cases with known expected scores
        test_cases = [
            {
                "response": "Looking at this code, I need to check if the array is sorted. <measurements>[true]</measurements>",
                "ground_truth": "[true]",
                "dataset_name": "function_correctness",
                "extra_info": {"is_correct": True},
                "expected_components": {
                    "format_reward": 1.0,  # Has <measurements> tag
                    "correctness_reward": 3.0,  # Correct prediction
                    "verbosity_reward": 0.0,  # No penalty expected
                    "monitor_correctness_reward": 0.0  # Monitor disabled
                }
            },
            {
                "response": "This function appears incorrect. <measurements>[false]</measurements>",
                "ground_truth": "[false]", 
                "dataset_name": "function_correctness",
                "extra_info": {"is_correct": False},
                "expected_components": {
                    "format_reward": 1.0,  # Has <measurements> tag
                    "correctness_reward": 3.0,  # Correct prediction  
                    "verbosity_reward": 0.0,  # No penalty expected
                    "monitor_correctness_reward": 0.0  # Monitor disabled
                }
            },
            {
                "response": "Wrong format without tags - true prediction",
                "ground_truth": "[true]",
                "dataset_name": "function_correctness", 
                "extra_info": {"is_correct": True},
                "expected_components": {
                    "format_reward": 0.0,  # Missing <measurements> tag
                    "correctness_reward": 0.0,  # Wrong format = no correctness reward
                    "verbosity_reward": 0.0,  # No penalty expected
                    "monitor_correctness_reward": 0.0  # Monitor disabled
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 Test Case {i+1}:")
            print(f"   Response: {test_case['response'][:50]}...")
            print(f"   Ground Truth: {test_case['ground_truth']}")
            
            result = compute_score(
                response=test_case["response"],
                ground_truth=test_case["ground_truth"],
                dataset_name=test_case["dataset_name"],
                extra_info=test_case["extra_info"]
            )
            
            # Verify result structure
            assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
            
            required_fields = [
                "score", "format_reward", "correctness_reward", 
                "verbosity_reward", "monitor_correctness_reward"
            ]
            for field in required_fields:
                assert field in result, f"Missing field: {field}"
            
            # Check individual components
            expected = test_case["expected_components"]
            for component, expected_value in expected.items():
                actual_value = result[component]
                assert actual_value == expected_value, \
                    f"Test case {i+1}: {component} = {actual_value}, expected {expected_value}"
                print(f"   ✅ {component}: {actual_value}")
            
            # Check total score
            expected_total = sum(expected.values())
            assert result["score"] == expected_total, \
                f"Test case {i+1}: total score = {result['score']}, expected {expected_total}"
            print(f"   ✅ Total score: {result['score']}")
        
        print(f"\n✅ All {len(test_cases)} single-call test cases passed!")
        
    finally:
        # Restore original monitor setting
        if original_monitor is None:
            os.environ.pop("TRAIN_WITH_MONITOR", None)
        else:
            os.environ["TRAIN_WITH_MONITOR"] = original_monitor


def test_non_monitor_rewards_batch_call():
    """
    Test that format, correctness, and verbosity rewards work in batch-call mode
    with monitor disabled.
    """
    print("\n" + "="*80)
    print("NON-MONITOR REWARDS TEST: BATCH CALL")
    print("="*80)
    
    # Disable monitor for this test
    original_monitor = os.environ.get("TRAIN_WITH_MONITOR")
    os.environ["TRAIN_WITH_MONITOR"] = "false"
    
    try:
        # Import AFTER setting environment
        from rewards import compute_score
        
        # Batch test data
        responses = [
            "Code analysis shows correctness. <measurements>[true]</measurements>",
            "Function has bugs. <measurements>[false]</measurements>", 
            "Missing format - true answer"
        ]
        ground_truths = ["[true]", "[false]", "[true]"]
        extra_infos = [
            {"is_correct": True},
            {"is_correct": False}, 
            {"is_correct": True}
        ]
        
        print(f"📊 Testing batch call with {len(responses)} samples")
        print("🚫 Monitor disabled - testing core rewards only")
        
        # Call with batch interface (as BatchRewardManager would)
        batch_scores = compute_score(
            data_sources=["function_correctness"] * len(responses),
            solution_strs=responses,
            ground_truths=ground_truths,
            extra_infos=extra_infos
        )
        
        # Verify batch results
        assert isinstance(batch_scores, list), f"Batch result should be list, got {type(batch_scores)}"
        assert len(batch_scores) == len(responses), f"Expected {len(responses)} scores, got {len(batch_scores)}"
        
        # Expected scores for each test case
        expected_scores = [
            4.0,  # format(1) + correctness(3) + verbosity(0) + monitor(0)
            4.0,  # format(1) + correctness(3) + verbosity(0) + monitor(0)  
            0.0   # format(0) + correctness(0) + verbosity(0) + monitor(0) - wrong format
        ]
        
        for i, (actual_score, expected_score) in enumerate(zip(batch_scores, expected_scores)):
            assert isinstance(actual_score, (int, float)), f"Score {i} should be numeric, got {type(actual_score)}"
            assert actual_score == expected_score, \
                f"Sample {i}: score = {actual_score}, expected {expected_score}"
            print(f"   ✅ Sample {i+1}: {actual_score} (expected {expected_score})")
        
        print(f"\n✅ Batch call test passed! All {len(responses)} samples scored correctly.")
        
        # Verify consistency with single calls
        print("\n🔄 Verifying consistency with single calls...")
        for i, (response, ground_truth, extra_info) in enumerate(zip(responses, ground_truths, extra_infos)):
            single_result = compute_score(
                response=response,
                ground_truth=ground_truth, 
                dataset_name="function_correctness",
                extra_info=extra_info
            )
            single_score = single_result["score"]
            batch_score = batch_scores[i]
            
            assert single_score == batch_score, \
                f"Sample {i}: single call score ({single_score}) != batch call score ({batch_score})"
            print(f"   ✅ Sample {i+1}: single={single_score}, batch={batch_score} ✓")
        
        print("✅ Single vs batch consistency verified!")
        
    finally:
        # Restore original monitor setting
        if original_monitor is None:
            os.environ.pop("TRAIN_WITH_MONITOR", None)
        else:
            os.environ["TRAIN_WITH_MONITOR"] = original_monitor


@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set - skipping monitor test"
)
def test_monitor_can_be_toggled():
    """
    Test that monitor rewards can be toggled on/off without breaking the system.
    """
    print("\n" + "="*80)
    print("MONITOR TOGGLE TEST")
    print("="*80)
    
    # Test sample
    response = "This code looks correct. <measurements>[true]</measurements>"
    ground_truth = "[true]"
    extra_info = {"is_correct": True}
    
    # Test with monitor OFF
    os.environ["TRAIN_WITH_MONITOR"] = "false"
    from rewards import compute_score
    
    result_no_monitor = compute_score(
        response=response,
        ground_truth=ground_truth,
        dataset_name="function_correctness", 
        extra_info=extra_info
    )
    
    print(f"🚫 Monitor OFF: score = {result_no_monitor['score']}")
    print(f"   Monitor reward: {result_no_monitor['monitor_correctness_reward']}")
    assert result_no_monitor['monitor_correctness_reward'] == 0.0, "Monitor reward should be 0 when disabled"
    
    # Test with monitor ON
    os.environ["TRAIN_WITH_MONITOR"] = "true"
    
    # Need to reimport to pick up new environment setting
    import importlib
    import rewards
    importlib.reload(rewards)
    from rewards import compute_score
    
    result_with_monitor = compute_score(
        response=response,
        ground_truth=ground_truth,
        dataset_name="function_correctness",
        extra_info=extra_info
    )
    
    print(f"✅ Monitor ON: score = {result_with_monitor['score']}")
    print(f"   Monitor reward: {result_with_monitor['monitor_correctness_reward']}")
    
    # With monitor, we should get additional reward
    assert result_with_monitor['score'] > result_no_monitor['score'], \
        "Score with monitor should be higher than without monitor"
    
    assert result_with_monitor['monitor_correctness_reward'] > 0, \
        "Monitor reward should be positive when enabled"
    
    print("✅ Monitor can be toggled successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
