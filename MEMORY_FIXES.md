# CUDA Out of Memory Error Fixes

This document outlines the fixes applied to resolve the `torch.OutOfMemoryError: CUDA out of memory` error.

## Root Causes Identified

1. **Missing tensor cleanup**: Audio tensors and model outputs were not being explicitly deleted from GPU memory between batches
2. **No GPU cache clearing**: `torch.cuda.empty_cache()` was not being called between processing batches
3. **Aggressive memory usage**: Batch size calculation was using 1/4 of GPU RAM, which left insufficient memory for processing overhead
4. **Memory fragmentation**: PyTorch's default memory allocation can cause fragmentation

## Fixes Applied

### 1. Memory Management in `identify_languages` Function
- Added explicit tensor deletion (`del input_features`, `del logits`, `del wavs`)
- Added `torch.cuda.empty_cache()` call after each batch
- Wrapped processing in try/finally block to ensure cleanup happens even on errors

### 2. Memory Management in `load_and_resample_wavs` Function
- Added explicit deletion of `raw_wav` tensors after resampling
- Added proper exception handling to prevent memory leaks on errors

### 3. Memory Management in `AIModel.transcribe` Method
- Added `torch.cuda.empty_cache()` call after each batch
- Wrapped batch processing in try/finally block

### 4. Reduced Batch Size
- Changed `calculate_batch_bytes()` to use 1/8 of GPU RAM instead of 1/4
- This provides more memory headroom for processing overhead

### 5. Environment Variable Configuration
- Created `run_transcription.sh` script that sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- This helps reduce memory fragmentation as recommended by PyTorch

### 6. Memory Monitoring
- Added `print_gpu_memory_usage()` function for debugging
- Added memory monitoring after each batch in the main processing loop

## Usage

To run the application with improved memory management:

```bash
# Use the shell script to set environment variables and run
./run_transcription.sh head --sftp_url "your_sftp_url"

# Or set the environment variable manually
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py head --sftp_url "your_sftp_url"
```

## Monitoring

The application now prints GPU memory usage after each batch:
```
GPU Memory - Allocated: 2.34 GB, Reserved: 2.86 GB, Total: 11.51 GB
```

This helps identify if memory usage is growing unexpectedly between batches.

## Additional Recommendations

If you still encounter memory issues:

1. **Further reduce batch size**: Edit `utils.py` and change `gpu_ram_bytes_cur // 8` to `gpu_ram_bytes_cur // 16`
2. **Process smaller files first**: The batching algorithm tries to fit files by total size, but very large individual files might still cause issues
3. **Monitor system memory**: Use `nvidia-smi` to monitor GPU usage during processing
4. **Restart periodically**: For very long-running jobs, consider processing in smaller chunks with periodic restarts 