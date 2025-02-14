#!/bin/bash

(
mkdir -p pretrained
cd pretrained || exit

# pretrained models for fine tuning on ImageNet
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt # Trained by OpenAI
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt # Trained by OpenAI

# Pretrained models for fine tuning CelebA-HQ
gdown https://drive.google.com/uc?id=1norNWWGYP3EZ_o05DmoW1ryKuKMmhlCX  # Trained by https://github.com/andreas128/RePaint
)
