#!/bin/bash

#SBATCH --output=logs/%x-%j_%A_%a.out
#SBATCH --error=logs/%x-%j_%A_%a.err
#SBATCH --mail-type=ALL

#SBATCH --job-name=generate
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=1


module purge
module load miniconda


HOST="localhost"
PORT="8000"
GPU_MEM_UTIL="0.95"
TENSOR_PAR_SIZE="1"
VLLM_LOG="vllm_server.log"


start_vllm() {
    MODEL="${1}"

    echo "starting vllm"
    conda activate vllm

    # start vllm server in the background
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    vllm serve --host "${HOST}" --port "${PORT}" \
        --gpu_memory_utilization "${GPU_MEM_UTIL}" \
        --tensor-parallel-size "${TENSOR_PAR_SIZE}" \
        "${MODEL}" >> "${VLLM_LOG}" 2>&1 &

    VLLM_PID=$!

    conda deactivate

    # wait for the vllm server to connect
    until $(nc -z "${HOST}" "${PORT}" 2> /dev/null)
    do
        if ! $(kill -0 "${VLLM_PID}" 2> /dev/null); then
            echo "vllm terminated abruptly"
            exit 1
        else
            echo "waiting for vllm to connect to ${HOST}:${PORT}"
            sleep 30
        fi
    done

    echo "vllm is connected"

    # return the process id of vllm as exit status
    return "${VLLM_PID}"
}

stop_vllm() {
    VLLM_PID="${1}"

    echo "shutting down vllm"

    kill -SIGTERM "${VLLM_PID}"
    wait "${VLLM_PID}"

    echo "vllm has shut down"
}


run_main() {
    # expects valid arguments of main.py, but
    # --model_id <str> should be the first two
    MODEL="${2}"

    # check if the model is supported
    if ! $(grep -q "model_id=\"${MODEL}\"" local_models.py); then
        echo "Unsupported model: ${MODEL}, not found in local_models.py"
        return 1
    fi

    start_vllm "${MODEL}"
    VLLM_PID=$!

    conda activate nlp
    python main.py "$@"

    stop_vllm "${VLLM_PID}"
    conda deactivate
}


#run_main --model_id "deepseek-ai/deepseek-coder-33b-instruct"
#run_main --model_id "Qwen/Qwen2.5-Coder-32B-Instruct"

#run_main --model_id "bigcode/starcoder2-15b-instruct-v0.1"
#run_main --model_id "codellama/CodeLlama-13b-Instruct-hf"

#run_main --model_id "NTQAI/Nxcode-CQ-7B-orpo"
#run_main --model_id "Artigenz/Artigenz-Coder-DS-6.7B"
