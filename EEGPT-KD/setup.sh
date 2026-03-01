#!/bin/bash
# Setup script for EEGPT-KD pipeline
# Clones EEGPT repo and provides checkpoint download instructions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"

mkdir -p "$VENDOR_DIR"

# Clone EEGPT if not already present
if [ ! -d "$VENDOR_DIR/EEGPT" ]; then
    echo "Cloning EEGPT repository..."
    git clone https://github.com/BINE022/EEGPT.git "$VENDOR_DIR/EEGPT"
else
    echo "EEGPT already cloned at $VENDOR_DIR/EEGPT"
fi

# Checkpoint download instructions
CKPT_DIR="$VENDOR_DIR/EEGPT/downstream/Modules/checkpoints"
mkdir -p "$CKPT_DIR"

if [ ! -f "$CKPT_DIR/eegpt_mcae_58chs_4s_large4E.ckpt" ]; then
    echo ""
    echo "=========================================="
    echo "CHECKPOINT DOWNLOAD REQUIRED"
    echo "=========================================="
    echo "Download the pretrained EEGPT checkpoint from Figshare:"
    echo "  https://figshare.com/articles/dataset/EEGPT/27270773"
    echo ""
    echo "Place the file at:"
    echo "  $CKPT_DIR/eegpt_mcae_58chs_4s_large4E.ckpt"
    echo "=========================================="
else
    echo "Checkpoint found at $CKPT_DIR/eegpt_mcae_58chs_4s_large4E.ckpt"
fi

echo "Setup complete."
