bigcodebench.syncheck --samples "${1}"

bigcodebench.evaluate --split complete --subset full --samples "${1}"
