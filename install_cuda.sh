#!/bin/bash
export LLAMA_CUBLAS=1
#export CMAKE_ARGS=-DLLAMA_CUBLAS=on
#export FORCE_CMAKE=1

source ~/anaconda3/bin/activate
#check if scrapalot-chat virtual env exists
if conda info --envs | grep -q "scrapalot-chat"
then
  echo "env already exists"
  conda activate /usr/local/anaconda3/envs/scrapalot-chat
else
  conda create -y -n "scrapalot-chat"
  conda activate /usr/local/anaconda3/envs/scrapalot-chat
  pip3 install -r requirements.txt
fi

echo "Done! Active envs:"
conda info --envs


# if you're not using conda
# check if scrapalot-chat exists
#if [ ! -d "scrapalot-chat" ]; then
#    python3 -m scrapalot-chat scrapalot-chat
#fi
#source scrapalot-chat/bin/activate
#pip install -r requirements.txt
