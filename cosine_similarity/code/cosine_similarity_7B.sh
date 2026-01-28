#!/bin/bash
MODEL_SIZE="7B"
SCRIPT_PATH=main/cosine_similarity/code/cosine_similarity_main.py

ACTIVATIONS_DIR=main/id_analysis/results/qwen2.5_${MODEL_SIZE}/activations

OUTPUT_DIR=main/cosine_similarity/results/qwen2.5_${MODEL_SIZE}

CONDA_ENV_PATH="" # path to your conda env
PYTHON_CMD="${CONDA_ENV_PATH}/bin/python"

echo "Starting Cosine Similarity Analysis..."
echo "Reading activations from: ${ACTIVATIONS_DIR}"

${PYTHON_CMD} ${SCRIPT_PATH} \
  --activations_dir ${ACTIVATIONS_DIR} \
  --output_dir ${OUTPUT_DIR}

echo "Done."