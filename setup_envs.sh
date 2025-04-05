module load miniconda

conda create -n vllm python=3.12 -y
conda activate vllm

pip install vllm
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
conda deactivate


conda create -n nlp python=3.12 -y
conda activate nlp

pip install langfun[all] --pre
conda deactivate

conda create -n bcb python=3.12 -y
conda activate bcb
pip install bigcodebench --upgrade
conda deactivate
