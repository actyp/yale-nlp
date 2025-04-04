# info: https://docs.ycrc.yale.edu/clusters/grace/
# constraint: --constraint a100-80g | a5000 | rtx3090
# check gpu: nvidia-smi

salloc -t 6:00:00 --partition gpu_devel --gpus 1
