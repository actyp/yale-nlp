module load miniconda

# vllm for the server
conda create -n vllm python=3.12 -y
conda activate vllm

pip install vllm==0.8.2
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
conda deactivate


# nlp for the langfun client
conda create -n nlp python=3.12 -y
conda activate nlp

pip install langfun[all]==0.1.2.dev202504180804
pip install autopep8==2.3.2
pip install datasets==3.5.0
pip install huggingface_hub[cli]==0.30.2
pip install matplotlib==3.10.1
conda deactivate

# bcb for the bigcodebench remote evaluation
conda create -n bcb python=3.12 -y
conda activate bcb

pip install bigcodebench==0.2.5
conda deactivate
