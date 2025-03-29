module load miniconda
conda activate vllm

model="Qwen/Qwen2.5-7B-Instruct"

vllm serve --gpu_memory_utilization 0.95 ${model}
