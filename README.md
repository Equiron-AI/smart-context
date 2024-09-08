## Environment
To create a Python virtual environment, use the command:
```console
conda env create -f environment.yml
```

## Installation
```console
pip install lmtools
```

## Supported Models
The following LLM models are supported:
- CohereForAI/c4ai-command-r-08-2024
- google/gemma-2-27b-it
- mistralai/Mistral-Large-Instruct-2407

## LLM backends
The following LLM backends are supported:
- Llama.cpp Server API

## Run Llama.CPP Server backend
```console
llama.cpp/build/bin/llama-server -m model_q5_k_m.gguf -ngl 99 -fa -c 4096 --host 0.0.0.0 --port 8000
```

## Install CUDA toolkit for Llama.cpp compilation
Please note that the toolkit version must match the driver version. The driver version can be found using the nvidia-smi command.
To install toolkit for CUDA 12.5 you need to run the following commands:
```console
CUDA_TOOLKIT_VERSION=12-5
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-${CUDA_TOOLKIT_VERSION}
echo -e '
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
' >> ~/.bashrc
```
