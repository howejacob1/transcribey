#!/bin/bash
pushd /home/bantaim/conserver/transcribey
source .venv/bin/activate
python main.py discover --production --url sftp://bantaim@banidk0:22/media/10900-hdd-0/
popd