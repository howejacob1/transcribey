#!/bin/bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --break-system-packages
pip install -r /home/bantaim/conserver/transcribey/ --break-system-packages --force-reinstall
pip install nemo_toolkit['all'] --break-system-packages --force-reinstall 

cd /home/bantaim/conserver/transcribey/
source venv/bin/activate 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
pip install -r /home/bantaim/conserver/transcribey/ --force-reinstall
pip install nemo_toolkit['all'] --force-reinstall
