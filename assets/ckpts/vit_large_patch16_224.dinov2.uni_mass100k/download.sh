#!/usr/bin/env bash

# Change to the script's directory
cd $(dirname "$0")
echo "Downloading data to '$(echo "${PWD}" | sed 's|'"${HOME}/"'||')'"

URL="https://huggingface.co/MahmoodLab/UNI/resolve/main/pytorch_model.bin"

if [[ -f "pytorch_model.bin" ]]; then
  echo "Model weights already exist. Skipping download."
  exit 0
fi

if [[ ! -f "${HOME}/.cache/huggingface/token" ]]; then
  printf "HuggingFace token not found. "
  echo "Visit https://huggingface.co/settings/tokens to create one."
else
  HF_TOKE=$(head -n1 ${HOME}/.cache/huggingface/token)
fi

echo "Download the model weights for the UNI2 model from the Mahmood Lab"

if [[ -z "${HF_TOKE}" ]]; then
  echo "HuggingFace token not found. Use 'huggingface-cli login' if needed."
  wget "${URL}"
else
  wget --header="Authorization: Bearer ${HF_TOKE}" "${URL}"
fi