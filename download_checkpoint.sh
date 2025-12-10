#!/bin/bash
set -e
python3 - << 'EOF'
from xvader.xvader_utils.load_weight import save_checkpoint
url = "hf://shun0705/xvader-vitl-depth/checkpoint.pt"
output_path = "checkpoints/checkpoint.pt"
save_checkpoint(url, output_path)
EOF