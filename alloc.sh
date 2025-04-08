# info: https://docs.ycrc.yale.edu/clusters/grace/
# check gpu: nvidia-smi

salloc -t 6:00:00 --partition gpu_devel --gpus 1 --constraint 'a100|a5000|rtx3090'
