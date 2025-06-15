#!/bin/bash

source .venv/bin/activate
COMMAND="python main.py head --dataset med"
echo $COMMAND
$COMMAND