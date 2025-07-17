#!/usr/bin/env python3
"""
WAV to OGG Vorbis converter using ffmpeg-python library.
Supports both local and remote operation modes.

Requirements:
- ffmpeg binary: sudo apt update && sudo apt install ffmpeg  
- ffmpeg-python library: pip install ffmpeg-python
- For remote mode: paramiko library for SFTP
"""
import argparse
import hashlib
import os
import queue
import random
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
try:
    import ffmpeg
except ImportError:
    print("ERROR: ffmpeg-python library not found!")
    print("Please install it with: pip install ffmpeg-python")
    exit(1)
# try PyAV first to avoid launching ffmpeg for every file
try:
    import av
    HAVE_PYAV = True
except ImportError:
    HAVE_PYAV = False

# Always use external ffmpeg – PyAV path is slower in practice
HAVE_PYAV = False

from mongo_utils import db, _db_semaphore
from utils import num_cores
from vcon_class import Vcon
import sftp

# Target directories to save converted files
TARGET_DRIVES = [
    "/media/500-ssd-0",
    "/media/500-hdd-0", 
    "/media/1800-hdd-0"
]

SOURCE_DRIVE = "/media/10900-hdd-0"

# Remote connection settings
REMOTE_HOST_CONFIG = "banidk0-remote"  # SSH config name
SFTP_URL = "sftp://bantaim@banidk0:22/"

# Minimum free space required (in bytes) - 1GB safety margin
MIN_FREE_SPACE = 1 * 1024 * 1024 * 1024  # 1GB

# Global variables for operation mode
IS_REMOTE_MODE = False
sftp_client = None
ssh_client = None
remote_temp_dir = None

# Global variable to store available drives (checked once at startup)
AVAILABLE_DRIVES = []

# Cached drive weights and update tracking
DRIVE_WEIGHTS = []
DRIVE_WEIGHTS_LOCK = threading.Lock()
LAST_WEIGHT_UPDATE = 0
WEIGHT_UPDATE_INTERVAL = 20000  # Update weights every 20000 files

# Thread-safe counter for progress tracking
progress_lock = threading.Lock()
total_processed = 0  # successfully converted files
# additional cumulative stats
total_corrupt = 0    # files marked as corrupt / failed
# cumulative counters
total_bytes_processed = 0  # successfully converted input bytes
# Track when the worker started converting for wall-clock throughput
worker_start_time = None

# Work queue for batch processing
work_queue = queue.Queue()
BATCH_SIZE = 3000  # Reserve 3000 vcons at a time

def cpu_cores_for_conversion():
    """Get optimal number of CPU cores for conversion tasks"""
    return num_cores()+5

def setup_remote_mode():
    """Initialize SFTP connection and temporary directory for remote operation"""
    global sftp_client, ssh_client, remote_temp_dir
    
    print(f"Setting up remote connection to {REMOTE_HOST_CONFIG}...")
    
    try:
        # Connect using existing sftp module
        sftp_client, ssh_client = sftp.connect(SFTP_URL)
        print("✓ SFTP connection established")
        
        # Create temporary directory for remote operations
        remote_temp_dir = tempfile.mkdtemp(prefix="transcribey_remote_")
        os.makedirs(remote_temp_dir, exist_ok=True)
        print(f"✓ Temporary directory: {remote_temp_dir}")
        
        # Test database connection
        db.find_one({"done": True}, {"_id": 1})
        print("✓ Database connection verified")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to setup remote mode: {e}")
        cleanup_remote_mode()
        return False

def cleanup_remote_mode():
    """Clean up SFTP connection and temporary files"""
    global sftp_client, ssh_client, remote_temp_dir
    
    print("Cleaning up remote mode...")
    
    # Clean up temporary directory
    if remote_temp_dir and os.path.exists(remote_temp_dir):
        try:
            shutil.rmtree(remote_temp_dir)
            print(f"✓ Removed temporary directory: {remote_temp_dir}")
        except Exception as e:
            print(f"WARNING: Could not remove temp dir {remote_temp_dir}: {e}")
    
    # Close SFTP connection
    if sftp_client:
        try:
            sftp_client.close()
            print("✓ SFTP connection closed")
        except:
            pass
    
    # Close SSH connection  
    if ssh_client:
        try:
            ssh_client.close()
            print("✓ SSH connection closed")
        except:
            pass
    
    sftp_client = None
    ssh_client = None
    remote_temp_dir = None

def get_hostname():
    """Get the current hostname to identify this worker"""
    return socket.gethostname()

def get_available_space(drive_path):
    """Get available space on a drive in bytes"""
    # Treat any path that exists locally as a local path, even when running in remote mode
    if os.path.exists(drive_path):
        try:
            _, _, free_bytes = shutil.disk_usage(drive_path)
            return free_bytes
        except Exception as e:
            print(f"WARNING: Could not check space for {drive_path}: {e}")
            return 0

    if IS_REMOTE_MODE:
        try:
            stat = sftp_client.statvfs(drive_path)
            free_bytes = stat.f_bavail * stat.f_frsize
            return free_bytes
        except Exception as e:
            print(f"WARNING: Could not check remote space for {drive_path}: {e}")
            return 0

    # Fallback – unknown path
    return 0

def initialize_available_drives():
    """Check drive space once at startup and cache available drives"""
    global AVAILABLE_DRIVES
    
    if IS_REMOTE_MODE:
        print("Remote mode: checking remote target drives via SFTP...")
        AVAILABLE_DRIVES = []
        
        for drive in TARGET_DRIVES:
            try:
                # Check if remote drive exists
                sftp_client.stat(drive)
                free_space = get_available_space(drive)
                
                if free_space >= MIN_FREE_SPACE:
                    print(f"OK: {drive}: {free_space / 1024 / 1024 / 1024:.2f} GB free (remote)")
                    AVAILABLE_DRIVES.append(drive)
                else:
                    print(f"ERROR: {drive}: {free_space / 1024 / 1024 / 1024:.2f} GB free (insufficient, remote)")
            except Exception as e:
                print(f"ERROR: Remote drive {drive} does not exist or is inaccessible: {e}")
                continue
        
        # Also check local temp space
        if remote_temp_dir:
            local_free = get_available_space(remote_temp_dir)
            print(f"Local temp space: {local_free / 1024 / 1024 / 1024:.2f} GB free")
            if local_free < MIN_FREE_SPACE:
                print("WARNING: Local temporary space is low!")
    else:
        # Local mode - existing behavior
        print("Checking target drives (one-time check)...")
        AVAILABLE_DRIVES = []
        
        for drive in TARGET_DRIVES:
            if not os.path.exists(drive):
                print(f"ERROR: {drive} does not exist")
                continue
                
            free_space = get_available_space(drive)
            
            if free_space >= MIN_FREE_SPACE:
                print(f"OK: {drive}: {free_space / 1024 / 1024 / 1024:.2f} GB free")
                AVAILABLE_DRIVES.append(drive)
            else:
                print(f"ERROR: {drive}: {free_space / 1024 / 1024 / 1024:.2f} GB free (insufficient)")
    
    if not AVAILABLE_DRIVES:
        print("ERROR: No drives have sufficient free space!")
        return False
    
    print(f"OK: Will use {len(AVAILABLE_DRIVES)} drive(s) for conversions")
    
    # Initialize drive weights cache
    update_drive_weights()
    
    return True

def update_drive_weights():
    """Update cached drive weights based on current free space"""
    global DRIVE_WEIGHTS, LAST_WEIGHT_UPDATE
    
    with DRIVE_WEIGHTS_LOCK:
        DRIVE_WEIGHTS = []
        for drive in AVAILABLE_DRIVES:
            free_space = get_available_space(drive)
            if free_space < MIN_FREE_SPACE:
                # Drive is full, give it zero weight
                DRIVE_WEIGHTS.append(0)
            else:
                DRIVE_WEIGHTS.append(free_space)
        
        LAST_WEIGHT_UPDATE = total_processed
        print(f"Updated drive weights: {[f'{w/1024/1024/1024:.1f}GB' for w in DRIVE_WEIGHTS]}")

def get_target_drive_by_hash(filename):
    """Distribute files across available target drives using weighted random selection based on cached free space"""
    if not AVAILABLE_DRIVES:
        return None
    
    # Check if we need to update weights
    if not DRIVE_WEIGHTS or (total_processed - LAST_WEIGHT_UPDATE) >= WEIGHT_UPDATE_INTERVAL:
        update_drive_weights()
    
    # Use cached weights for selection
    with DRIVE_WEIGHTS_LOCK:
        current_weights = DRIVE_WEIGHTS.copy()
    
    # If all drives are full, return None
    if sum(current_weights) == 0:
        return None
    
    # Weighted random selection - drives with more free space have higher probability
    return random.choices(AVAILABLE_DRIVES, weights=current_weights)[0]

def ensure_target_directory_exists(target_path):
    """Create target directory if it doesn't exist"""
    if IS_REMOTE_MODE:
        # For remote mode, create directory via SFTP
        target_dir = os.path.dirname(target_path)
        try:
            # Try to create directory recursively via SFTP
            dirs_to_create = []
            current_dir = target_dir
            
            # Find which directories need to be created
            while current_dir and current_dir != '/':
                try:
                    sftp_client.stat(current_dir)
                    break  # Directory exists, stop here
                except:
                    dirs_to_create.append(current_dir)
                    current_dir = os.path.dirname(current_dir)
            
            # Create directories from parent to child
            for dir_path in reversed(dirs_to_create):
                try:
                    sftp_client.mkdir(dir_path)
                except:
                    pass  # Directory might already exist
                    
        except Exception as e:
            print(f"WARNING: Could not create remote directory {target_dir}: {e}")
    else:
        # Local mode - existing behavior
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)

def convert_wav_to_ogg(source_path, target_path):
    try:
        ensure_target_directory_exists(target_path)

        if HAVE_PYAV:
            # fast in-process conversion
            in_container = av.open(source_path, "r")
            out_container = av.open(target_path, "w", format="ogg")

            out_stream = out_container.add_stream("libopus", rate=8000, options={"vbr":"on", "compression_level":"10"})
            out_stream.bit_rate = 8000  # 8 kbps
            # make stream mono if supported
            try:
                out_stream.codec_context.layout = "mono"
            except AttributeError:
                # older PyAV versions use channel_layout property
                try:
                    out_stream.codec_context.channel_layout = "mono"
                except AttributeError:
                    pass  # fallback, encoder will infer from frames

            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=8000)

            for in_frame in in_container.decode(audio=0):
                resampled_frames = resampler.resample(in_frame)
                # resampler may return a list of frames or a single frame
                if not isinstance(resampled_frames, (list, tuple)):
                    resampled_frames = [resampled_frames]

                for r_frame in resampled_frames:
                    for pkt in out_stream.encode(r_frame):
                        out_container.mux(pkt)

            # flush encoder
            for pkt in out_stream.encode(None):
                out_container.mux(pkt)

            out_container.close()
            in_container.close()
            return True, None

        # fallback to ffmpeg-python external process
        stream = ffmpeg.input(source_path)
        stream = ffmpeg.output(
            stream,
            target_path,
            ar=8000,
            ac=1,
            acodec="libopus",
            **{"b:a": "8k", "vbr": "on", "compression_level": "10"},
            f="opus",
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        return True, None

    except ffmpeg.Error as e:
        # ffmpeg-python raised an error or PyAV not available
        return False, str(e)
    except Exception as e:
        # catch-all, including any PyAV errors
        return False, str(e)

def reserve_batch_vcons():
    """Reserve a batch of vcons atomically and add them to work queue"""
    query = {
        "done": True,
        "corrupt": {"$ne": True},
        "converting": {"$ne": True},  # Not already being converted
        "dialog.0.filename": {
            # Pick up source recordings that are still WAV or already OGG
            "$regex": f"^{SOURCE_DRIVE}/.*\\.(wav|ogg)$",
            "$options": "i"
        }
    }
    
    update = {"$set": {"converting": True}}
    
    reserved_count = 0
    
    with _db_semaphore:
        # Find and mark vcons as being converted atomically
        for _ in range(BATCH_SIZE):
            vcon_dict = db.find_one_and_update(
                query,
                update,
                return_document=True
            )
            
            if vcon_dict:
                vcon = Vcon.from_dict(vcon_dict)
                # Check if file exists before adding to queue
                file_exists = False
                if IS_REMOTE_MODE:
                    try:
                        sftp_client.stat(vcon.filename)
                        file_exists = True
                    except:
                        file_exists = False
                else:
                    file_exists = os.path.exists(vcon.filename)
                
                if file_exists:
                    work_queue.put(vcon)
                    reserved_count += 1
                else:
                    # Mark as corrupt if file doesn't exist
                    db.update_one(
                        {"_id": vcon.uuid},
                        {"$set": {"corrupt": True, "converting": False}}
                    )
            else:
                # No more vcons available
                break
    
    if reserved_count > 0:
        print(f"Reserved {reserved_count} vcons for processing")
    
    return reserved_count

def get_vcon_from_queue():
    """Get a vcon from the work queue (non-blocking)"""
    try:
        return work_queue.get_nowait()
    except queue.Empty:
        return None

def update_vcon_filename(vcon, new_filename):
    """Update the vcon filename in the database and clear converting flag"""
    try:
        with _db_semaphore:
            result = db.update_one(
                {"_id": vcon.uuid},
                {"$set": {"dialog.0.filename": new_filename}, "$unset": {"converting": ""}}
            )
        
        if result.modified_count > 0:
            return True
        else:
            print(f"ERROR: Failed to update database (no documents modified)")
            # Clear converting flag even if update failed
            db.update_one({"_id": vcon.uuid}, {"$unset": {"converting": ""}})
            return False
            
    except Exception as e:
        print(f"ERROR: Error updating database: {e}")
        # Clear converting flag on error
        try:
            db.update_one({"_id": vcon.uuid}, {"$unset": {"converting": ""}})
        except:
            pass
        return False

def process_one_vcon():
    """Process one vcon: convert WAV to OGG, update database, remove original"""
    
    vcon = get_vcon_from_queue()
    
    if not vcon:
        # No work in queue
        return False
    
    original_filename = vcon.filename
    
    if IS_REMOTE_MODE:
        return process_one_vcon_remote(vcon, original_filename)
    else:
        return process_one_vcon_local(vcon, original_filename)

def process_one_vcon_local(vcon, original_filename):
    """Process one vcon in local mode (existing behavior)"""
    
    # Check if original file exists
    if not os.path.exists(original_filename):
        print(f"ERROR: Original file does not exist: {original_filename}")
        # Clear converting flag and mark as corrupt
        try:
            with _db_semaphore:
                db.update_one(
                    {"_id": vcon.uuid},
                    {"$set": {"corrupt": True}, "$unset": {"converting": ""}}
                )
        except:
            pass
        record_corrupt()
        return False
    
    # Generate new filename path
    original_path = Path(original_filename)
    relative_path = original_path.relative_to(SOURCE_DRIVE)
    
    # Change extension to .opus
    new_relative_path = relative_path.with_suffix('.opus')
    
    # Choose target drive from cached available drives
    target_drive = get_target_drive_by_hash(str(relative_path))
    if not target_drive:
        print("ERROR: No available drives for conversion")
        record_corrupt()
        return False
        
    new_filename = os.path.join(target_drive, str(new_relative_path))
    
    # show converting message with stats
    with progress_lock:
        elapsed = time.time() - worker_start_time if worker_start_time else 0
        throughput = (total_bytes_processed / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
        stats_str = f"{total_processed} {total_corrupt} {throughput:.2f} MB/s"
    print(f"{stats_str} {new_filename[5:]}")

    # Measure conversion time for stats
    file_size_bytes = os.path.getsize(original_filename)
    start_t = time.time()
    success, error_msg = convert_wav_to_ogg(original_filename, new_filename)
    duration = time.time() - start_t
    
    if not success:
        print(f"ERROR: Conversion failed: {error_msg}")
        # Clear converting flag and mark as corrupt
        try:
            with _db_semaphore:
                db.update_one(
                    {"_id": vcon.uuid},
                    {"$set": {"corrupt": True}, "$unset": {"converting": ""}}
                )
        except:
            pass
        record_corrupt()
        return False
    
    # Check new file exists
    if not os.path.exists(new_filename):
        print(f"ERROR: Converted file not found: {new_filename}")
        record_corrupt()
        return False
    
    # Update database
    if not update_vcon_filename(vcon, new_filename):
        print("ERROR: Database update failed, keeping original file")
        try:
            os.remove(new_filename)
            print("Removed converted file due to database failure")
        except:
            pass
        record_corrupt()
        return False
    
    # Remove original file
    try:
        os.remove(original_filename)
    except Exception as e:
        pass  # Ignore removal errors
    
    # record stats for successful conversion
    record_success(file_size_bytes, duration)
    return True

def process_one_vcon_remote(vcon, original_filename):
    """Process one vcon in remote mode: download, convert, upload"""
    
    local_wav_path = None
    local_ogg_path = None
    
    try:
        # Check if remote original file exists
        try:
            sftp_client.stat(original_filename)
        except Exception as e:
            print(f"ERROR: Remote original file does not exist: {original_filename}")
            record_corrupt()
            # Clear converting flag and mark as corrupt
            try:
                with _db_semaphore:
                    db.update_one(
                        {"_id": vcon.uuid},
                        {"$set": {"corrupt": True}, "$unset": {"converting": ""}}
                    )
            except:
                pass
            return False
        
        # Generate paths
        original_path = Path(original_filename)
        relative_path = original_path.relative_to(SOURCE_DRIVE)
        new_relative_path = relative_path.with_suffix('.opus')
        
        # Choose target drive
        target_drive = get_target_drive_by_hash(str(relative_path))
        if not target_drive:
            print("ERROR: No available drives for conversion")
            record_corrupt()
            return False
        
        remote_new_filename = os.path.join(target_drive, str(new_relative_path))
        
        # Create local temporary file paths
        local_wav_path = os.path.join(remote_temp_dir, f"temp_{vcon.uuid}.wav")
        local_ogg_path = os.path.join(remote_temp_dir, f"temp_{vcon.uuid}.ogg")
        
        # Download WAV file from remote
        print(f"Downloading {os.path.basename(original_filename)} [{get_hostname()}]")
        sftp.download_optimized(original_filename, local_wav_path, sftp_client)
        
        # show converting message with stats
        with progress_lock:
            elapsed = time.time() - worker_start_time if worker_start_time else 0
            throughput = (total_bytes_processed / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0
            stats_str = f"{total_processed} files, {total_corrupt} corrupt, {throughput:.2f} MB/s"
        print(f"Converting ({stats_str}) {remote_new_filename} [{get_hostname()}]")

        # Measure conversion time for stats
        file_size_bytes = os.path.getsize(local_wav_path)
        start_t = time.time()
        success, error_msg = convert_wav_to_ogg(local_wav_path, local_ogg_path)
        duration = time.time() - start_t
        
        if not success:
            print(f"ERROR: Conversion failed: {error_msg}")
            record_corrupt()
            # Clear converting flag and mark as corrupt
            try:
                with _db_semaphore:
                    db.update_one(
                        {"_id": vcon.uuid},
                        {"$set": {"corrupt": True}, "$unset": {"converting": ""}}
                    )
            except:
                pass
            return False
        
        # Check local converted file exists
        if not os.path.exists(local_ogg_path):
            print(f"ERROR: Local converted file not found: {local_ogg_path}")
            record_corrupt()
            return False
        
        # Ensure the destination directory exists on the remote host before upload
        ensure_target_directory_exists(remote_new_filename)

        # Upload OGG file to remote
        print(f"Uploading {os.path.basename(remote_new_filename)} [{get_hostname()}]")
        sftp_client.put(local_ogg_path, remote_new_filename)
        
        # Verify remote file exists
        try:
            sftp_client.stat(remote_new_filename)
        except Exception as e:
            print(f"ERROR: Remote converted file not found after upload: {remote_new_filename}")
            record_corrupt()
            return False
        
        # Update database
        if not update_vcon_filename(vcon, remote_new_filename):
            print("ERROR: Database update failed, cleaning up")
            record_corrupt()
            try:
                sftp_client.remove(remote_new_filename)
                print("Removed remote converted file due to database failure")
            except:
                pass
            return False
        
        # Remove original remote file
        try:
            sftp_client.remove(original_filename)
        except Exception as e:
            pass  # Ignore removal errors
        
        # Successful conversion – record stats
        record_success(file_size_bytes, duration)
        return True
        
    finally:
        # Clean up local temporary files
        for temp_file in [local_wav_path, local_ogg_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def increment_progress():
    # this definition is kept for backward compatibility but should not be used
    # in the new flow.  It simply returns the current processed count without
    # printing anything.
    with progress_lock:
        return total_processed


# NEW: record successful conversion and print stats line
def record_success(size_bytes: int, duration_sec: float):
    """Update global counters for a successful conversion and print a stats line."""
    global total_processed, total_bytes_processed, worker_start_time

    with progress_lock:
        total_processed += 1
        total_bytes_processed += size_bytes

        # no printing here; stats are shown in the upcoming 'Converting' line


# NEW: record corrupt / failed file and print stats line
def record_corrupt():
    """Increment corrupt counter and print current stats line."""
    global total_corrupt, worker_start_time

    with progress_lock:
        total_corrupt += 1

        # no printing here; stats are shown in the upcoming 'Converting' line


# Updated worker – stats are handled inside the processing functions
def process_vcon_worker():
    """Worker function for thread pool - processes one vcon"""
    process_one_vcon()
    return None

def cleanup_converting_flags():
    """Clear any remaining converting flags on exit for this worker only"""
    # Note: We don't clear ALL converting flags as other workers may still be processing
    # Only clear flags for items that were in our local work queue
    queued_items = []
    
    # Drain any remaining items from our work queue
    while True:
        try:
            vcon = work_queue.get_nowait()
            queued_items.append(vcon.uuid)
        except queue.Empty:
            break
    
    # Clear converting flags for items we had reserved but didn't process
    if queued_items:
        try:
            with _db_semaphore:
                result = db.update_many(
                    {"_id": {"$in": queued_items}, "converting": True},
                    {"$unset": {"converting": ""}}
                )
                if result.modified_count > 0:
                    print(f"Cleaned up {result.modified_count} converting flags for this worker")
        except Exception as e:
            print(f"WARNING: Error cleaning up: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="WAV to OGG Vorbis converter with local and remote support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode (default)
  python convert_to_ogg.py

  # Remote mode
  python convert_to_ogg.py --remote

  # Remote mode with custom SSH host
  python convert_to_ogg.py --remote --host other-server-remote
        """)
    
    parser.add_argument('--remote', action='store_true',
                      help='Run in remote mode (download/upload files via SFTP)')
    parser.add_argument('--host', default=REMOTE_HOST_CONFIG,
                      help=f'SSH config hostname for remote mode (default: {REMOTE_HOST_CONFIG})')
    parser.add_argument('--sftp-url', 
                      help=f'Custom SFTP URL (default: {SFTP_URL})')
    
    return parser.parse_args()

def main():
    global total_processed, IS_REMOTE_MODE, REMOTE_HOST_CONFIG, SFTP_URL
    
    # Parse command line arguments
    args = parse_arguments()
    IS_REMOTE_MODE = args.remote
    REMOTE_HOST_CONFIG = args.host
    if args.sftp_url:
        SFTP_URL = args.sftp_url
    else:
        # Re-create the SFTP URL using the supplied host while keeping the existing username & port
        url_body = SFTP_URL.split("://")[1]
        username, host_and_port = url_body.split("@")
        # host_and_port is like "oldhost:22/" – we keep the port chunk after ':'
        if ":" in host_and_port:
            port_part = host_and_port.split(":")[1]
        else:
            port_part = "22/"
        SFTP_URL = f"sftp://{username}@{REMOTE_HOST_CONFIG}:{port_part}"
    
    total_processed = 0  # Reset counters
    global total_corrupt, total_bytes_processed, worker_start_time
    total_corrupt = 0
    total_bytes_processed = 0
    worker_start_time = time.time()
    
    # Get optimal number of threads based on CPU cores
    max_workers = cpu_cores_for_conversion()
    
    mode_str = "REMOTE" if IS_REMOTE_MODE else "LOCAL"
    hostname = get_hostname()
    
    print(f"WAV to OGG Vorbis Converter for VCONs ({max_workers} Threads) - {mode_str} MODE")
    print(f"Worker: {hostname}")
    print("=" * 70)
    
    if IS_REMOTE_MODE:
        print("Remote mode - this worker will:")
        print("1. Connect to remote server via SFTP")
        print("2. Find done vcons with WAV files (in remote database)")
        print("3. Download WAV files to local temporary directory")
        print("4. Convert WAV to OGG Vorbis format locally")
        print("5. Upload OGG files back to remote server")
        print("6. Update the remote database")
        print("7. Remove the original remote WAV file")
        print("8. Clean up local temporary files")
        print("9. Repeat until no more WAV files found")
    else:
        print("Local mode - this script will:")
        print("1. Find done vcons with WAV files")
        print("2. Check available disk space on target drives") 
        print(f"3. Convert WAV to OGG Vorbis format ({max_workers} parallel threads)")
        print("4. Save to a drive with sufficient free space")
        print("5. Update the database")
        print("6. Remove the original WAV file")
        print("7. Repeat until no more WAV files found")
    print()
    
    # Check if ffmpeg is available
    try:
        # Test that ffmpeg-python can access ffmpeg by checking version
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("OK: ffmpeg binary is available")
        print("OK: ffmpeg-python library is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg binary is not available. Please install it first:")
        print("   sudo apt update && sudo apt install ffmpeg")
        return False
    
    # Initialize remote mode if requested
    if IS_REMOTE_MODE:
        print()
        if not setup_remote_mode():
            return False
    
    # Initialize available drives (one-time check)
    print()
    if not initialize_available_drives():
        if IS_REMOTE_MODE:
            cleanup_remote_mode()
        return False
    
    # Check source drive exists
    if IS_REMOTE_MODE:
        try:
            sftp_client.stat(SOURCE_DRIVE)
            print(f"OK: Remote source drive {SOURCE_DRIVE} exists")
        except Exception as e:
            print(f"ERROR: Remote source drive {SOURCE_DRIVE} does not exist: {e}")
            cleanup_remote_mode()
            return False
    else:
        if os.path.exists(SOURCE_DRIVE):
            print(f"OK: Source drive {SOURCE_DRIVE} exists")
        else:
            print(f"ERROR: Source drive {SOURCE_DRIVE} does not exist")
            return False
    
    print()
    print(f"Starting parallel conversion process with {max_workers} threads...")
    print("   Press Ctrl+C to stop at any time")
    print()
    
    try:
        # Reserve initial batch of work
        print("Reserving initial batch of vcons...")
        reserve_batch_vcons()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit initial batch of work
            futures = []
            for _ in range(max_workers):
                future = executor.submit(process_vcon_worker)
                futures.append(future)
            
            while True:
                # Check if we need to reserve more work
                if work_queue.qsize() < max_workers:
                    reserved = reserve_batch_vcons()
                    if reserved == 0:
                        print("INFO: No more vcons available to reserve")
                
                # Check completed futures and submit new work
                new_futures = []
                active_work = False
                
                for future in futures:
                    if future.done():
                        try:
                            result = future.result()
                            # Always submit new work regardless of result
                            new_future = executor.submit(process_vcon_worker)
                            new_futures.append(new_future)
                        except Exception as e:
                            print(f"WARNING: Thread error: {e}")
                            # Submit replacement work
                            new_future = executor.submit(process_vcon_worker)
                            new_futures.append(new_future)
                    else:
                        # Keep the future that's still running
                        new_futures.append(future)
                        active_work = True
                
                futures = new_futures
                
                # If no active work and no work in queue, we're done
                if not active_work and work_queue.empty():
                    # Wait a bit for any remaining futures to complete
                    all_done = True
                    for future in futures:
                        if not future.done():
                            all_done = False
                            break
                    
                    if all_done:
                        break
                
                # Brief pause
                time.sleep(0.5)
                    
    except KeyboardInterrupt:
        print(f"\n\nWARNING: Interrupted by user (Ctrl+C) on {get_hostname()}")
        print(f"Processed {total_processed} files before interruption")
    
    # Clean up any remaining converting flags
    print("\nCleaning up...")
    cleanup_converting_flags()
    
    # Clean up remote mode if active
    if IS_REMOTE_MODE:
        cleanup_remote_mode()
    
    print("\n" + "=" * 70)
    print(f"CONVERSION COMPLETE! (Worker: {get_hostname()})")
    print(f"Total files processed: {total_processed}")
    
    if total_processed == 0:
        print("INFO: No WAV files found to convert. All done!")
    else:
        mode_str = "remote" if IS_REMOTE_MODE else "local"
        print(f"Successfully converted {total_processed} WAV files to OGG format ({mode_str} mode)")
        print("Database updated and original WAV files removed")
    
    print("=" * 70)
    return total_processed > 0

if __name__ == "__main__":
    main() 