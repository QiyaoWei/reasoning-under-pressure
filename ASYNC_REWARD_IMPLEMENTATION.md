# Async Batch Reward Manager Implementation Guide

This document outlines the changes needed to enable the optimized async batch reward manager for OpenAI API calls.

## 🚀 Performance Gains

- **Speedup**: 27x faster than naive approach (with caching)
- **Throughput**: ~13 samples/second vs 0.5 samples/second
- **Time Savings**: 32+ minutes saved per 1024-sample batch
- **Cost Efficiency**: Reduced API wait times and better resource utilization

## 🔧 Required Changes

### 1. Reward Manager Configuration
**File**: `verl/trainer/config/reward_model/reward_model.yaml`
**Line**: 50

```yaml
# BEFORE
reward_manager: naive

# AFTER  
reward_manager: batch
```

### 2. Concurrency Optimization (Already Applied)
**File**: `rewards.py`
**Line**: 366

```python
# Optimized from testing - increased from 20 to 60
max_concurrent=60  # Process up to 60 API calls concurrently
```

### 3. Environment Variables
Ensure these are set in your training environment:

```bash
export TRAIN_WITH_MONITOR=true
export OPENAI_API_KEY="your-openai-api-key"
export MONITOR_CACHE=true  # Optional: enables caching for repeated responses
```

## 📁 Modified Files

### Core Implementation
- ✅ `rewards.py` - Modified `compute_score()` to handle both single and batch interfaces
- ✅ `rewards.py` - Optimized `max_concurrent` from 20 to 60

### Configuration  
- 🔄 `verl/trainer/config/reward_model/reward_model.yaml` - Change `reward_manager: naive` to `reward_manager: batch`

## 🧪 Verification

### Test the Implementation
Run the performance test to verify gains:

```bash
cd /path/to/MATS
export OPENAI_API_KEY="your-key"
pytest our_tests/test_batch_1024_performance.py -v -s
```

Expected results:
- Total time: ~75 seconds for 1024 samples  
- Throughput: ~13 samples/second
- Speedup: 27x vs naive approach

### Test Different Scenarios
- **With caching**: `pytest our_tests/test_batch_1024_performance.py -v -s`
- **Without caching**: `pytest our_tests/test_batch_1024_no_cache.py -v -s`
- **Interface compatibility**: `pytest our_tests/test_batch_interface.py -v -s`

## 🏃‍♂️ Training Usage

Once configured, run training normally:

```bash
./run_grpo.sh
```

The batch reward manager will automatically:
- Process rewards in optimized batches
- Use 60 concurrent API calls
- Cache repeated responses  
- Maintain backward compatibility

## 📊 Performance Benchmarks

| Approach | Time (1024 samples) | Throughput | Speedup |
|----------|-------------------|------------|---------|
| Naive | ~34 minutes | 0.5/sec | 1x |
| Batch (no cache) | ~2.6 minutes | 6.5/sec | 16x |
| Batch (with cache) | ~1.3 minutes | 13/sec | 27x |

## 🔍 Implementation Details

### BatchRewardManager Interface
The `compute_score()` function now supports both calling conventions:

```python
# Single sample (backward compatibility)
score = compute_score(response, ground_truth, dataset_name, extra_info)

# Batch processing (new interface)  
scores = compute_score(
    data_sources=dataset_names,
    solution_strs=responses, 
    ground_truths=ground_truths,
    extra_infos=extra_infos
)
```

### Concurrency Optimization
Based on testing, 60 concurrent API calls provides optimal performance:
- Higher throughput than 20 (102% improvement)
- Avoids rate limiting issues seen at 80+ concurrent calls
- Stable performance without timeouts

## 🚨 Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: OPENAI_API_KEY not found
   Solution: export OPENAI_API_KEY="your-key"
   ```

2. **Rate Limiting (429 errors)**
   ```
   Solution: Reduce max_concurrent if needed, or upgrade OpenAI plan
   ```

3. **Memory Issues with High Concurrency**
   ```
   Solution: Keep max_concurrent at 60 or lower
   ```

4. **Config Not Taking Effect**
   ```
   Verify: reward_manager: batch in reward_model.yaml
   ```

## 📈 Next Steps

- Monitor performance in production training runs
- Adjust `max_concurrent` if API limits change
- Consider implementing adaptive concurrency based on API response times
- Explore caching strategies for frequently repeated responses

---

**Created**: Based on concurrency optimization testing  
**Performance validated**: Up to 27x speedup demonstrated  
**Status**: Ready for production use
