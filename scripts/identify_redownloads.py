import os
import time
import mongo_utils

all_filenames_path = "all-filenames.txt"
output_path = "files_to_download_2.txt"

start_time = time.time()
total_bytes_processed = 0
bytes_to_redownload = 0

# Ensure output file is empty at start
with open(output_path, "w") as _:
    pass

collection = mongo_utils.db

with open(all_filenames_path, "r") as f_all, open(output_path, "a") as f_out:
    for line in f_all:
        line = line.strip()
        if not line:
            continue

        first_space = line.find(" ")
        if first_space == -1:
            continue

        size_str = line[:first_space]
        foreign_path = line[first_space + 1 :]

        try:
            foreign_size = int(size_str)
        except ValueError:
            continue

        basename_full = os.path.basename(foreign_path)
        basename_only = os.path.splitext(basename_full)[0]

        # Find vcon by basename in first dialog entry
        vcon_doc = collection.find_one({"basename": basename_only}, {"done": 1, "corrupt": 1, "dialog": 1})

        need_redownload = False

        if vcon_doc is None:
            # No vcon – need redownload
            need_redownload = True
        else:
            is_done = vcon_doc.get("done", False)
            is_corrupt = vcon_doc.get("corrupt", False)

            if is_corrupt:
                need_redownload = True
            else:
                if is_done:
                    # Done and not corrupt – skip
                    need_redownload = False
                else:
                    # Compare file size if local file exists
                    dialog = vcon_doc.get("dialog", [])
                    local_path = None
                    if dialog and isinstance(dialog, list):
                        local_path = dialog[0].get("filename")
                    if local_path and os.path.exists(local_path):
                        try:
                            local_size = os.path.getsize(local_path)
                        except OSError:
                            local_size = -1
                    else:
                        local_size = -1

                    if local_size != foreign_size:
                        need_redownload = True

        # Update cumulative bytes processed (for data rate calculation)
        total_bytes_processed += foreign_size
        elapsed = time.time() - start_time
        mb_per_sec = (total_bytes_processed / 1048576) / elapsed if elapsed else 0

        if need_redownload:
            f_out.write(f"{foreign_path}\n")
            print(f"must redownload {basename_only} {mb_per_sec:.2f} MB/s")
        else:
            print(f"skipping {basename_only} {mb_per_sec:.2f} MB/s") 