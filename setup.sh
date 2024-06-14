#!/bin/bash

# Function to check if a file exists and return 0 or 1
check_model() {
  if [ -f "$1" ]; then
    return 0
  else
    return 1
  fi
}

# Display installed models
echo "Following models are installed, [x] means installed:"

check_model "./src/lm_finetune/nllb/model/nllb200densedst1bcheckpoint.pt"
nllb_installed=$?
if [ $nllb_installed -eq 0 ]; then echo "[x] NLLB"; else echo "[ ] NLLB"; fi

check_model "./src/lm_finetune/m2m100/model/1.2B_last_checkpoint.pt"
m2m100_installed=$?
if [ $m2m100_installed -eq 0 ]; then echo "[x] M2M100"; else echo "[ ] M2M100"; fi

check_model "./src/lm_finetune/small100/model/model_small100_fairseq.pt"
small100_installed=$?
if [ $small100_installed -eq 0 ]; then echo "[x] SMaLL100"; else echo "[ ] SMaLL100"; fi

# Create an array for choices
choices=()

if [ $nllb_installed -ne 0 ]; then choices+=("1) NLLB"); fi
if [ $m2m100_installed -ne 0 ]; then choices+=("2) M2M100"); fi
if [ $small100_installed -ne 0 ]; then choices+=("3) SMaLL100"); fi

if [ ${#choices[@]} -eq 0 ]; then
  echo "All models are already installed."
else
  echo "Select models to install (space-separated list):"
  for choice in "${choices[@]}"; do
    echo "$choice"
  done
  read -p "Enter your choice: " -a user_choices

  # Function to download a model
  download_model() {
    if [ ! -f "$2" ]; then
      echo "Downloading $(basename $2) ..."
      wget -P "$3" "$1"
      mv "$3/$(basename $1)" "$2"
    fi
  }

  # Process user choices
  for choice in "${user_choices[@]}"; do
    case $choice in
      1)
        download_model "https://tinyurl.com/nllb200densedst1bcheckpoint" "./src/lm_finetune/nllb/model/nllb200densedst1bcheckpoint.pt" "./src/lm_finetune/nllb/model"
        ;;
      2)
        download_model "https://dl.fbaipublicfiles.com/m2m_100/1.2B_last_checkpoint.pt" "./src/lm_finetune/m2m100/model/1.2B_last_checkpoint.pt" "./src/lm_finetune/m2m100/model"
        ;;
      3)
        # Download from amazon s3
        download_model "https://proxylmbucket.s3.us-east-2.amazonaws.com/model_small100_fairseq.pt" "./src/lm_finetune/small100/model/model_small100_fairseq.pt" "./src/lm_finetune/small100/model"
        ;;
      *)
        echo "Invalid choice: $choice"
        ;;
    esac
  done
fi

# Download COMET models
if [ ! -f "./src/lm_finetune/metrics/comet/model/wmt22-comet-da.ckpt" ]; then
    echo "Downloading COMET models ..."
    wget -P ./src/lm_finetune/metrics/comet/model https://huggingface.co/Unbabel/wmt22-comet-da/resolve/main/checkpoints/model.ckpt
    mv ./src/lm_finetune/metrics/comet/model/model.ckpt ./src/lm_finetune/metrics/comet/model/wmt22-comet-da.ckpt
fi

# Downloading xlm-roberta-large model
if [ ! -f "./src/lm_finetune/metrics/comet/xlm-roberta-large-model/pytorch_model.bin" ]; then
    echo "Downloading xlm-roberta-large model ..."
    wget -P ./src/lm_finetune/metrics/comet/xlm-roberta-large-model https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/pytorch_model.bin
fi

# Download sacrebleu spm
if [ ! -f "~/.sacrebleu/models/flores200sacrebleuspm" ]; then
    echo "Downloading sacrebleu spm ..."
    mkdir -p ~/.sacrebleu/models
    wget -P ~/.sacrebleu/models/ https://tinyurl.com/flores200sacrebleuspm
fi

# Install requirements
echo "Installing requirements ..."
pip install -r requirements.txt > /dev/null 2>&1 && echo "Requirements installed successfully."

# Installing fairseq
cd src/lm_finetune
if [ ! -d "fairseq" ]; then
    echo "Cloning fairseq repository ..."
    git clone https://github.com/ritsukkiii/fairseq.git
else
    echo "Skipping git clone, fairseq already exists."
fi
cd fairseq
echo "Installing fairseq ..."
pip install --editable ./ > /dev/null 2>&1 && echo "fairseq installed successfully."
echo "Installing additional fairseq dependencies ..."
pip install sentencepiece sacrebleu tensorboardX > /dev/null 2>&1 && echo "Additional fairseq dependencies installed successfully."

# Check dataset
cd ../../../
if [ ! -d "./experiments/dataset" ]; then
  read -p "Dataset not found. Do you want to download it? (y/n): " download_dataset
  download_dataset=${download_dataset:-y}
  if [ "$download_dataset" == "y" ]; then
    echo "Downloading dataset ..."
    mkdir -p ./experiments
    wget -P ./experiments https://proxylmbucket.s3.us-east-2.amazonaws.com/dataset.tar.gz
    echo "Unzipping dataset ..."
    cd ./experiments
    tar -xzvf dataset.tar.gz dataset
    cd ../
  else
    echo "Skipping dataset download."
  fi
else
  echo "Dataset already exists."
fi
