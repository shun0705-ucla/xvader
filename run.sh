#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/vggt"
export PYTHONPATH="$PYTHONPATH:$(pwd)/third_party/Depth_Anything_V2"
export PYTHONPATH="$PYTHONPATH:$(pwd)/xvader"


IMG_PATH="./image"
REF_PATH="./ref"
OUTDIR="./output"
ENCODER="${4:-vitl}"  # Default to vitl if not specified

python3 run.py \
    --img_path "$IMG_PATH" \
    --ref_path "$REF_PATH" \
    --outdir "$OUTDIR" \
    --encoder "$ENCODER"