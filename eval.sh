#!/bin/bash

module purge
module load miniconda
conda activate bcb

SAMPLES="${1}"
SAMPLES_NUM="1140"

SANITIZE_SUCCESS="Sanitized ${SAMPLES_NUM} out of ${SAMPLES_NUM} files."
SYNCHECK_SUCCESS="All ${SAMPLES_NUM} code are compilable!"

EVAL_LOG="${1}_eval.log"

echo "Sanitizing ${SAMPLES}"
bigcodebench.sanitize --calibrate True --samples "${SAMPLES}" 2>"${EVAL_LOG}" | tee "${EVAL_LOG}" | grep "${SANITIZE_SUCCESS}"
[[ $! -eq 0 ]] || { echo "Failed sanitizing ${SAMPLES}"; exit 1; }

SAN_SAMPLES="${SAMPLES/.jsonl/-sanitized-calibrated.jsonl}"
[[ -f "${SAN_SAMPLES}" ]] || { echo "Failed to create ${SAN_SAMPLES}"; exit 1; }

echo "Synchecking ${SAN_SAMPLES}"
bigcodebench.syncheck --samples "${SAN_SAMPLES}" 2> "${EVAL_LOG}" | tee "${EVAL_LOG}" | grep "${SYNCHECK_SUCCESS}"
[[ $! -eq 0 ]] || { echo "Failed synchecking ${SAN_SAMPLES}"; exit 1; }

echo "Evaluating ${SAN_SAMPLES}"
bigcodebench.evaluate --split complete --subset full --samples "${SAN_SAMPLES}"
[[ $! -eq 0 ]] || { echo "Failed evaluating ${SAN_SAMPLES}"; exit 1; }

echo "Evaluated successfully ${SAN_SAMPLES}" && rm "${EVAL_LOG}"
