# Reward Optimization Testing Infrastructure

This directory contains essential tests for the async batch reward manager optimization. 

## 🧪 Minimal Test Structure

### `benchmarks/`
- `test_concurrency_optimization.py` - Find optimal concurrency settings for future tuning

### `integration/`
- `test_batch_interface.py` - Regression test for BatchRewardManager interface
- `test_non_monitor_rewards.py` - Test core rewards (format, correctness, verbosity) work without monitor

### `data/`
- `sample_responses_expanded.txt` - 482 test responses (~650 words each)

## 🚀 Running Tests

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Run regression tests
pytest our_tests/integration/test_batch_interface.py -v -s
pytest our_tests/integration/test_non_monitor_rewards.py -v -s

# Re-optimize concurrency if needed
pytest our_tests/benchmarks/test_concurrency_optimization.py -v -s
```

## 📈 Findings:

- throughput increases until max_concurrent = 1024 and starts decreasing by max_concurrent = 2048
- so setting max_concurrent to 1024/(number of simultaneous runs) will be safe and effective

## 🔧 Implementation

The optimized implementation requires only:
1. Modified `rewards.py` with batch interface support
2. Config change: set reward manager to batch (should be backwards compatible with existing reward code for all reward functions)

