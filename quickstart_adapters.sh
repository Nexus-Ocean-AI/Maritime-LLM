#!/bin/bash

# Quick start script for LoRA adapter upload and merge
# This script helps you upload and merge your trained LoRA adapters

set -e  # Exit on error

echo "=============================================================================="
echo "Maritime LLM - LoRA Adapter Upload & Merge Quick Start"
echo "=============================================================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're on the remote server or local machine
if [ -d "qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628" ]; then
    echo -e "${GREEN}✓${NC} Checkpoint directories found locally"
    CHECKPOINTS_AVAILABLE=true
else
    echo -e "${YELLOW}⚠${NC}  Checkpoint directories not found locally"
    echo "    Expected: qwen-maritime-longcontext-cpt/Phase_1a_Short/checkpoint-8628"
    echo "    Expected: qwen-maritime-longcontext-cpt/Phase_1b_Medium/checkpoint-872"
    echo ""
    echo "Options:"
    echo "  1. Run this script on the server where checkpoints exist"
    echo "  2. Copy checkpoints to local machine first"
    echo ""
    CHECKPOINTS_AVAILABLE=false
fi

echo ""
echo "What would you like to do?"
echo ""
echo "  1) Upload adapters to HuggingFace Hub"
echo "  2) Merge adapters with base model (locally)"
echo "  3) Merge and upload to HuggingFace"
echo "  4) Setup HuggingFace authentication"
echo "  5) Install dependencies"
echo "  6) Exit"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "=============================================================================="
        echo "Uploading Adapters to HuggingFace Hub"
        echo "=============================================================================="
        echo ""
        
        if [ "$CHECKPOINTS_AVAILABLE" = false ]; then
            echo -e "${RED}✗${NC} Checkpoints not available. Please ensure checkpoint directories exist."
            exit 1
        fi
        
        # Check if script is configured
        if grep -q "YOUR_HF_USERNAME" upload_adapters_to_hf.py; then
            echo -e "${YELLOW}⚠${NC}  Please configure your HuggingFace username first:"
            echo "    Edit upload_adapters_to_hf.py"
            echo "    Change: HF_USERNAME = 'YOUR_HF_USERNAME'"
            echo "    To:     HF_USERNAME = 'your_actual_username'"
            echo ""
            read -p "Press Enter after configuring..."
        fi
        
        echo "Running upload script..."
        python upload_adapters_to_hf.py
        ;;
        
    2)
        echo ""
        echo "=============================================================================="
        echo "Merging Adapters with Base Model"
        echo "=============================================================================="
        echo ""
        
        if [ "$CHECKPOINTS_AVAILABLE" = false ]; then
            echo -e "${RED}✗${NC} Checkpoints not available. Please ensure checkpoint directories exist."
            exit 1
        fi
        
        echo "Which adapter(s) to merge?"
        echo "  1) Phase 1a only"
        echo "  2) Phase 1b only"
        echo "  3) Both (all)"
        echo ""
        read -p "Enter choice (1-3): " merge_choice
        
        case $merge_choice in
            1)
                echo "Merging Phase 1a..."
                python merge_lora_adapters.py --adapter phase1a
                ;;
            2)
                echo "Merging Phase 1b..."
                python merge_lora_adapters.py --adapter phase1b
                ;;
            3)
                echo "Merging all adapters..."
                python merge_lora_adapters.py --adapter all
                ;;
            *)
                echo -e "${RED}✗${NC} Invalid choice"
                exit 1
                ;;
        esac
        ;;
        
    3)
        echo ""
        echo "=============================================================================="
        echo "Merge and Upload to HuggingFace"
        echo "=============================================================================="
        echo ""
        
        if [ "$CHECKPOINTS_AVAILABLE" = false ]; then
            echo -e "${RED}✗${NC} Checkpoints not available. Please ensure checkpoint directories exist."
            exit 1
        fi
        
        # Check if script is configured
        if grep -q "YOUR_HF_USERNAME" merge_lora_adapters.py; then
            echo -e "${YELLOW}⚠${NC}  Please configure your HuggingFace username first:"
            echo "    Edit merge_lora_adapters.py"
            echo "    Change: HF_USERNAME = 'YOUR_HF_USERNAME'"
            echo "    To:     HF_USERNAME = 'your_actual_username'"
            echo ""
            read -p "Press Enter after configuring..."
        fi
        
        echo "Merging and uploading all adapters..."
        python merge_lora_adapters.py --adapter all --upload
        ;;
        
    4)
        echo ""
        echo "=============================================================================="
        echo "HuggingFace Authentication Setup"
        echo "=============================================================================="
        echo ""
        echo "You need a HuggingFace account and access token."
        echo ""
        echo "Steps:"
        echo "  1. Create account at: https://huggingface.co/join"
        echo "  2. Get token at: https://huggingface.co/settings/tokens"
        echo "  3. Choose authentication method below"
        echo ""
        echo "Authentication methods:"
        echo "  1) Login via CLI (recommended)"
        echo "  2) Set environment variable"
        echo ""
        read -p "Enter choice (1-2): " auth_choice
        
        case $auth_choice in
            1)
                echo ""
                echo "Running HuggingFace CLI login..."
                huggingface-cli login
                ;;
            2)
                echo ""
                read -p "Enter your HuggingFace token: " hf_token
                echo ""
                echo "Add this to your shell profile (~/.bashrc or ~/.zshrc):"
                echo "  export HF_TOKEN='$hf_token'"
                echo ""
                export HF_TOKEN="$hf_token"
                echo -e "${GREEN}✓${NC} Token set for current session"
                ;;
            *)
                echo -e "${RED}✗${NC} Invalid choice"
                exit 1
                ;;
        esac
        ;;
        
    5)
        echo ""
        echo "=============================================================================="
        echo "Installing Dependencies"
        echo "=============================================================================="
        echo ""
        echo "Installing required packages..."
        pip install -q huggingface_hub transformers peft torch accelerate bitsandbytes
        echo ""
        echo -e "${GREEN}✓${NC} Dependencies installed"
        echo ""
        echo "Installed packages:"
        echo "  - huggingface_hub: Upload/download from HF Hub"
        echo "  - transformers: Load and run models"
        echo "  - peft: LoRA adapter support"
        echo "  - torch: PyTorch framework"
        echo "  - accelerate: Multi-GPU support"
        echo "  - bitsandbytes: Quantization support"
        ;;
        
    6)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo -e "${RED}✗${NC} Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=============================================================================="
echo -e "${GREEN}✓${NC} Done!"
echo "=============================================================================="
echo ""
echo "For more information, see: ADAPTER_UPLOAD_MERGE_GUIDE.md"
echo ""
