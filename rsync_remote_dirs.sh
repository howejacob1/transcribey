#!/usr/bin/env bash

REMOTE_USER="bantaim"
REMOTE_HOST="10.0.0.7"
REMOTE_BASE="/home/bantaim/conserver"
LOCAL_BASE="/home/jhowe/conserver"

DIRS=(
  "fake_wavs"
  "fake_wavs_cute"
  "fake_wavs_medlarge"
  "fake_wavs_mids"
  "openslr-12"
  "fake_wavs_large"
)

for DIR in "${DIRS[@]}"; do
  echo "Syncing $DIR..."
  rsync -avz --progress "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${DIR}/" "${LOCAL_BASE}/${DIR}/"
  if [ $? -eq 0 ]; then
    echo "Successfully synced $DIR."
  else
    echo "Error syncing $DIR."
  fi
done

echo "All done." 