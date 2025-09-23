"""
Test to find optimal concurrency level for OpenAI API calls.
Tests different max_concurrent values to find the sweet spot.

Run with:
    OPENAI_API_KEY=your-key pytest our_tests/test_concurrency_optimization.py -v -s
"""

import pytest
import os
import time
import sys
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.asyncio
async def test_concurrency_optimization():
    """
    Test different concurrency levels to find optimal performance.
    """
    print("\n" + "="*80)
    print("CONCURRENCY OPTIMIZATION TEST")
    print("="*80)

    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        pytest.skip("OPENAI_API_KEY not set")

    # Enable monitor and disable cache for accurate measurement
    os.environ["TRAIN_WITH_MONITOR"] = "true"
    os.environ["MONITOR_CACHE"] = "false"

    # Import rewards AFTER setting environment variables
    from rewards import compute_monitor_rewards_batch, initialize_openai_clients, monitor_cache

    # Ensure clients are initialized
    initialize_openai_clients()
    monitor_cache.clear()

    # Load sample data from the data directory
    responses_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_responses_expanded.txt")
    
    with open(responses_file, 'r') as f:
        sample_responses = [line.strip() for line in f.readlines() if line.strip()]

    print("📊 Testing with 10x max_concurrent samples for each level")
    print("🚫 Cache disabled for accurate measurement")
    print("\n" + "="*80)

    # Test different concurrency levels up to 150
    concurrency_levels = [10, 64, 128, 256, 512, 1024]
  
    results = []

    for max_concurrent in concurrency_levels:
        # Use 10x the concurrency level as sample count
        num_samples = 10 * max_concurrent
        test_responses = sample_responses[:num_samples]
        test_ground_truths = ["[true]"] * len(test_responses)
        test_extra_infos = [{"is_correct": True}] * len(test_responses)
        print(f"\n🔄 Testing max_concurrent = {max_concurrent}")
        print("-" * 40)
        
        # Clear cache between tests
        monitor_cache.clear()
        
        start_time = time.time()
        
        try:
            # Get event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the batch processing with this concurrency level
            # Add timeout for very high concurrency that might hang
            timeout_seconds = 120 if max_concurrent <= 100 else 180
            
            try:
                _, _, _ = await asyncio.wait_for(
                    compute_monitor_rewards_batch(
                        test_responses,
                        "function_correctness",
                        test_extra_infos,
                        max_concurrent=max_concurrent
                    ),
                    timeout=timeout_seconds
                )
                
                end_time = time.time()
                total_time = end_time - start_time
            except asyncio.TimeoutError:
                end_time = time.time()
                total_time = end_time - start_time
                raise Exception(f"Timeout after {timeout_seconds}s (actual time: {total_time:.1f}s)")
            
            throughput = len(test_responses) / total_time
            avg_time_per_call = total_time / len(test_responses)
            
            results.append({
                'max_concurrent': max_concurrent,
                'total_time': total_time,
                'throughput': throughput,
                'avg_time_per_call': avg_time_per_call,
                'success': True
            })
            
            # Estimate time for 1024 samples
            estimated_1024_time = 1024 / throughput
            print(f"   ✅ Success: {total_time:.2f}s total, {throughput:.2f} calls/sec")
            print(f"      📈 Estimated 1024-batch time: {estimated_1024_time:.1f}s ({estimated_1024_time/60:.1f} min)")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                'max_concurrent': max_concurrent,
                'total_time': float('inf'),
                'throughput': 0,
                'avg_time_per_call': float('inf'),
                'success': False,
                'error': str(e)
            })

    # Analyze results
    print("\n" + "="*80)
    print("CONCURRENCY OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"{'Concurrency':<12} {'Time (s)':<10} {'Throughput':<12} {'Status':<10} {'Speedup':<8}")
    print("-" * 60)
    
    baseline_time = None
    best_throughput = 0
    best_concurrency = None
    
    for result in results:
        if result['success']:
            if baseline_time is None:
                baseline_time = result['total_time']
                speedup = "1.00x"
            else:
                speedup = f"{baseline_time / result['total_time']:.2f}x"
            
            if result['throughput'] > best_throughput:
                best_throughput = result['throughput']
                best_concurrency = result['max_concurrent']
            
            status = "✅ OK"
            throughput_str = f"{result['throughput']:.2f}/sec"
        else:
            status = "❌ FAIL"
            speedup = "-"
            throughput_str = "0.00/sec"
        
        print(f"{result['max_concurrent']:<12} {result['total_time']:<10.2f} {throughput_str:<12} {status:<10} {speedup:<8}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if best_throughput > 0:
        print(f"🎯 OPTIMAL CONCURRENCY: {best_concurrency}")
        print(f"🚀 BEST THROUGHPUT: {best_throughput:.2f} calls/second")
        
        # Calculate improvement over current setting (20)
        current_result = next((r for r in results if r['max_concurrent'] == 20 and r['success']), None)
        if current_result and best_concurrency != 20:
            improvement = (best_throughput / current_result['throughput'] - 1) * 100
            print(f"📈 IMPROVEMENT over current (20): {improvement:.1f}% faster")
        
        # Practical recommendations based on actual results
        print(f"💡 Recommended: Use {best_concurrency} concurrent calls for optimal performance")
            
        print("\n🔧 To apply this change, modify the max_concurrent parameter in:")
        print("   rewards.py, line 367: max_concurrent=20  →  max_concurrent={best_concurrency}")
        
    else:
        print("❌ No successful results - API may be experiencing issues")

    print("\n" + "="*80)

    # Return best concurrency for any automated processing
    return {'best_concurrency': best_concurrency, 'best_throughput': best_throughput}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
