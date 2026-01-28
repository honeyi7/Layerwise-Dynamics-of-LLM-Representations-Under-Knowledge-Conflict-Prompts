#!/bin/bash

PARTITION=
WORKSPACE=
CONTAINER=
MOUNT=

jobname=causal_trace_qwen2.5_7B_$(date +%Y%m%d_%H%M%S)

NNODES=1
GPUS_PER_NODE=1

SCRIPT=main/causal_trace/code/causal_trace_main.py

SPEC=N6lS.Iu.I80.${GPUS_PER_NODE}

MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
KNOW_DATA=main/knowledge_probe/result/qwen2.5_7B/known_bucket.jsonl
UNKNOWN_DATA=main/knowledge_probe/result/qwen2.5_7B/unknown_bucket.jsonl
AMBIGUOUS_DATA=main/knowledge_probe/result/qwen2.5_7B/ambiguous_bucket.jsonl

INTERVENTION_KIND=state  # attn mlp state

OUTPUT_DIR=main/causal_trace/results/qwen2.5_7B/${INTERVENTION_KIND}
MAX_SAMPLES=-1
BATCH_SIZE=100
SAVE_INTERVAL=100

CONDA_ENV_PATH="/path/to/your/conda/envs"

ENV_SETUP="export PATH=${CONDA_ENV_PATH}/bin:\$PATH && \
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:\$LD_LIBRARY_PATH && \
export PYTHONPATH=${CONDA_ENV_PATH}/lib/python3.13/site-packages:\$PYTHONPATH"

PYTHON_CMD="${CONDA_ENV_PATH}/bin/python"

INSTALL_DEPS="${PYTHON_CMD} -m pip install accelerate --user"

MKDIR_CMD="mkdir -p ${OUTPUT_DIR}"

MAIN_CMD="${PYTHON_CMD} ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --know_data_path ${KNOW_DATA} \
  --unknown_data_path ${UNKNOWN_DATA} \
  --ambiguous_data_path ${AMBIGUOUS_DATA} \
  --output_dir ${OUTPUT_DIR} \
  --intervention_kind ${INTERVENTION_KIND} \
  --max_samples_per_dataset ${MAX_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --save_interval ${SAVE_INTERVAL}"

FULL_COMMAND="${ENV_SETUP} && ${MKDIR_CMD} && ${INSTALL_DEPS} && ${MAIN_CMD}"

echo "======================================"
echo "Job: ${jobname}"
echo "Model: 7B"
echo "Batch size: ${BATCH_SIZE}"
echo "======================================"

sco acp jobs create \
  --workspace-name $WORKSPACE \
  --aec2-name $PARTITION \
  --container-image-url $CONTAINER \
  --storage-mount $MOUNT \
  --training-framework pytorch \
  --worker-spec $SPEC \
  --worker-nodes $NNODES \
  -j $jobname \
  --command="${FULL_COMMAND}"

echo ""
echo "Submitted: ${jobname}"
echo "Check logs: sco acp jobs logs --workspace-name $WORKSPACE -j $jobname"