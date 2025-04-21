module load miniconda

# vllm for the server
conda create -n vllm python=3.12 -y
conda activate vllm

pip install vllm
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
conda deactivate


# nlp for the langfun client
conda create -n nlp python=3.12 -y
conda activate nlp

pip install langfun[all] --pre
pip install datasets
pip install autopep8
pip install -U "huggingface_hub[cli]"
conda deactivate

# bcb for the bigcodebench remote evaluation
conda create -n bcb python=3.12 -y
conda activate bcb

pip install bigcodebench --upgrade
conda deactivate
