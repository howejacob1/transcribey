#!/bin/bash
pushd /home/bantaim/conserver/transcribey/
source .venv/bin/activate
python main.py stats --production
deactivate
popd
