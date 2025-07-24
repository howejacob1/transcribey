# S3 Missing Files Downloader

This script downloads audio files from Digital Ocean Spaces buckets (vol1-eon through vol8-eon) that don't exist as vcons in your MongoDB database.

## Features

- **Smart Comparison**: Compares S3 objects with MongoDB vcons using the basename field (`dialog[0].basename`)
- **Intelligent Download Logic**: 
  - Downloads files missing from database
  - Downloads corrupt vcons only if local file differs from S3
  - Skips files that are done and not corrupt
  - Automatically unmarks corrupt vcons after successful download
- **Efficient Processing**: Processes files in batches to avoid overwhelming MongoDB
- **Checksum Verification**: Tracks file checksums to skip re-downloading identical files
- **Parallel Downloads**: Uses multiple threads for faster downloads
- **Resume Support**: Can be run multiple times - skips existing files with matching sizes/checksums
- **Progress Tracking**: Shows detailed progress and download statistics

## Setup

### 1. Install s5cmd (automatic)
The script will automatically install s5cmd from the included .deb file if not already available.

### 2. Test Public Access
The Digital Ocean Spaces buckets are configured as public, so no credentials are needed! 
Test the public access:

```bash
python3 setup_s3_credentials.py
```

This will verify that you can access the vol1-eon through vol8-eon buckets without credentials.

### 3. Test the complete setup
```bash
python3 test_setup.py
```

## Usage

### Run the Download Script
```bash
python3 download_missing_s3_files.py
```

The script will:
1. List all audio files from vol1-eon through vol8-eon buckets
2. Extract basenames from S3 object keys
3. Check which basenames don't exist in MongoDB (`dialog[0].basename`)
4. Download missing files to `/media/10900-hdd-0/` with proper directory structure
5. Skip files that already exist locally with matching sizes/checksums

### Directory Structure
Files are downloaded maintaining the S3 structure:
```
S3: vol1-eon/Freeswitch1/2025-07-03/15/file.wav
→ Local: /media/10900-hdd-0/Freeswitch1/2025-07-03/15/file.wav
```

## Configuration

You can modify these settings in `download_missing_s3_files.py`:

- **BATCH_SIZE**: Number of files processed per batch (default: 10,000)
- **MAX_WORKERS**: Number of parallel download threads (default: 8)
- **LOCAL_BASE_DIR**: Download destination (default: `/media/10900-hdd-0`)

## Progress and Monitoring

The script provides detailed progress information:
- Files processed vs total
- Download rate (files/second)
- Files downloaded vs skipped
- Batch processing status

Example output:
```
Processing batch 1/142 (10000 files)...
  Files to download: 2,847
  Skipping (exist in DB): 7,153
    Downloaded 10/2847 files...
  Progress: 10000/1420000 (0.7%) - 45.2 files/sec
```

## Resume and Reliability

- **Automatic Resume**: Run the script multiple times safely
- **Checksum Verification**: Files with matching sizes are assumed identical
- **Error Handling**: Individual download failures don't stop the entire process
- **Timeout Protection**: 5-minute timeout per download operation

## Troubleshooting

### Connection Issues
```bash
python3 setup_s3_credentials.py
```

### MongoDB Issues
Ensure your MongoDB connection is working:
```bash
python3 -c "from mongo_utils import db; print(f'MongoDB connected: {db.count_documents({})} vcons')"
```

### s5cmd Issues
Manually check s5cmd:
```bash
s5cmd --endpoint-url https://nyc3.digitaloceanspaces.com --no-sign-request ls
```

### Disk Space
Ensure sufficient space in `/media/10900-hdd-0/`:
```bash
df -h /media/10900-hdd-0/
```

## Performance Notes

- Processes millions of files efficiently using batched MongoDB queries
- Uses s5cmd for optimal S3 performance
- Maintains checksum cache to avoid redundant operations
- Designed for long-running operations with frequent progress updates

## Files Created

- **`/tmp/s3_checksums.json`**: Checksum cache (automatically managed)
- **`~/.aws/credentials`**: AWS credentials file (for s5cmd)

## Example Run

```bash
python3 download_missing_s3_files.py
```

```
S3 Missing Files Downloader
==================================================
✓ s5cmd already available: s5cmd version 2.3.0
Listing S3 objects from all buckets...
  Listing objects from vol1-eon...
    Found 234,567 audio files in vol1-eon
  Listing objects from vol2-eon...
    Found 345,678 audio files in vol2-eon
...
Total audio files found: 2,456,789

Processing 2,456,789 S3 objects...

Processing batch 1/246 (10000 files)...
  Files to download: 3,247
  Skipping (exist in DB): 6,753
    Downloaded 100/3247 files...
    Downloaded 200/3247 files...
...

==================================================
Summary:
  Total files processed: 2,456,789
  Files downloaded: 1,234,567
  Files skipped (already exist): 1,222,222
  Total time: 7,234.5 seconds
  Average rate: 339.5 files/sec
``` 