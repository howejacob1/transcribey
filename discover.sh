#!/bin/bash
pushd /home/bantaim/conserver/transcribey
source .venv/bin/activate
python main.py discover --production --url /media/10900-hdd-0/
popd