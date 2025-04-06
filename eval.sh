#!/bin/bash

module purge
module load miniconda
conda activate bcb

SAMPLES="${1}"
SAMPLES_NUM="1140"

SANITIZE_SUCCESS="Sanitized ${SAMPLES_NUM} out of ${SAMPLES_NUM} files."
SYNCHECK_SUCCESS="All ${SAMPLES_NUM} code are compilable!"

set -e

# sanitize first
echo "Sanitizing ${SAMPLES}"
bigcodebench.sanitize --calibrate True --samples "${SAMPLES}" 2>/dev/null | grep "${SANITIZE_SUCCESS}"


SAN_SAMPLES="${SAMPLES/.jsonl/-sanitized-calibrated.jsonl}"
echo "Synchecking ${SAN_SAMPLES}"
bigcodebench.syncheck --samples "${SAN_SAMPLES}" 2> /dev/null | grep "${SYNCHECK_SUCCESS}"

echo "Evaluating ${SAN_SAMPLES}"
bigcodebench.evaluate --split complete --subset full --samples "${SAN_SAMPLES}"

set +e
