#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt


read -p "Please provide the direct link to the model you'd like to use (Default https://huggingface.co/caffeinatedwoof/llama-2-7b-chat-hf-amod-mental-health-counseling-conversations-GGML/resolve/main/llama-2-7b-chat-hf-amod-mental-health-counseling-conversations.ggmlv3.q4_0.bin
):" MODELURL

MODELURL=${MODELURL:-https://huggingface.co/caffeinatedwoof/llama-2-7b-chat-hf-amod-mental-health-counseling-conversations-GGML/resolve/main/llama-2-7b-chat-hf-amod-mental-health-counseling-conversations.ggmlv3.q4_0.bin}

echo "Downloading $MODELURL"

wget -O ./llama_cpp_docker/modelfile.bin "$MODELURL"

echo -n "Enter the profile name you wish to use with cdk deploy (or just press Enter to use the default profile): "
read profile_name

if [ -z "$profile_name" ]; then
    cdk deploy
else
    cdk deploy --profile $profile_name
fi
