import subprocess
import time

ip = "45.55.123.39"
user = "root"

cmd = [
    "ssh",
    f"{user}@{ip}",
    "find /mnt -type f -name '*.wav' -printf '%s %p\\n'"
]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)

out = open("all-filenames.txt", "w")

start = time.time()
bytes_seen = 0
files_seen = 0

for line in proc.stdout:
    line = line.rstrip("\n")
    parts = line.split(" ", 1)
    if len(parts) != 2:
        continue
    size = int(parts[0])
    path = parts[1]

    out.write(f"{size} {path}\n")

    bytes_seen += size
    files_seen += 1
    elapsed = time.time() - start
    mb_per_sec = (bytes_seen / 1048576) / elapsed if elapsed else 0

    print(f"{path} {mb_per_sec:.2f}MB/s")

proc.wait()
out.close() 