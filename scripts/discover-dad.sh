#!/bin/bash

pushd /home/bantaim/conserver/transcribey/
source .venv/bin/activate
python main.py --production discover --url sftp://root@45.55.123.39:22/mnt/
popd