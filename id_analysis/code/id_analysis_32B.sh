#!/bin/bash

PARTITION=
WORKSPACE=
CONTAINER=
MOUNT=

jobname=id_analysis_qwen32B_$(date +%Y%m%d_%H%M%S)

NNODES=1
GPUS_PER_NODE=8


SCRIPT=main/id_analysis/code/id_analysis_main_32B.py

SPEC=N6lS.Iu.I80.${GPUS_PER_NODE}

MODEL_PATH="/path/to/Qwen2.5-${MODEL_SIZE}-Instruct"
PROBE_BASE="main/knowledge_probe/result/qwen2.5_${MODEL_SIZE}"

KNOW_DATA="${PROBE_BASE}/known_bucket.jsonl"
UNKNOWN_DATA="${PROBE_BASE}/unknown_bucket.jsonl"
AMBIGUOUS_DATA="${PROBE_BASE}/ambiguous_bucket.jsonl"

OUTPUT_DIR="main/id_analysis/results/qwen2.5_${MODEL_SIZE}"

MAX_SAMPLES=2000
BATCH_SIZE=32

CONDA_ENV_PATH="/path/to/your/conda/envs"
PYTHON_CMD="${CONDA_ENV_PATH}/bin/python"
INSTALL_DEPS="${PYTHON_CMD} -m pip install scikit-learn scipy seaborn matplotlib accelerate --user"

read -r -d '' RUN_SCRIPT << EOM
${ENV_SETUP}
export PATH=${CONDA_ENV_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:\$LD_LIBRARY_PATH

mkdir -p ${OUTPUT_DIR}
${INSTALL_DEPS}

echo "Starting ID Analysis Job..."

# -------------------------------------------------------
# Phase 1: Parallel Extraction (4 Workers x 2 GPUs)
# -------------------------------------------------------
NUM_WORKERS=4
GPUS_PER_WORKER=2
pids=()

for (( i=0; i<\$NUM_WORKERS; i++ ))
do
    start_gpu=\$(( i * GPUS_PER_WORKER ))
    end_gpu=\$(( start_gpu + GPUS_PER_WORKER - 1 ))
    export CUDA_VISIBLE_DEVICES="\${start_gpu},\${end_gpu}"
    
    echo "Launching Extraction Worker \$i on GPUs \$CUDA_VISIBLE_DEVICES"
    
    nohup ${PYTHON_CMD} ${SCRIPT} \\
      --step extract \\
      --model_path ${MODEL_PATH} \\
      --know_data_path ${KNOW_DATA} \\
      --unknown_data_path ${UNKNOWN_DATA} \\
      --ambiguous_data_path ${AMBIGUOUS_DATA} \\
      --output_dir ${OUTPUT_DIR} \\
      --max_samples_per_dataset ${MAX_SAMPLES} \\
      --batch_size ${BATCH_SIZE} \\
      --worker_id \$i \\
      --num_workers \$NUM_WORKERS \\
      > ${OUTPUT_DIR}/extract_worker_\${i}.log 2>&1 &
      
    pids+=(\$!)
done

echo "Waiting for extraction workers..."
for pid in \${pids[*]}; do
    wait \$pid
done
echo "Extraction finished."

# -------------------------------------------------------
# Phase 2: Analysis & Plotting (Single Process)
# -------------------------------------------------------
echo "Starting Analysis Phase (Concatenate & Compute ID)..."

export CUDA_VISIBLE_DEVICES=0 
${PYTHON_CMD} ${SCRIPT} \\
  --step analyze \\
  --output_dir ${OUTPUT_DIR}

echo "Job Completed. Check plots in ${OUTPUT_DIR}/plots"
EOM

echo "Submitting Job: ${jobname}"
sco acp jobs create \
  --workspace-name $WORKSPACE \
  --aec2-name $PARTITION \
  --container-image-url $CONTAINER \
  --storage-mount $MOUNT \
  --training-framework pytorch \
  --worker-spec $SPEC \
  --worker-nodes $NNODES \
  -j $jobname \
  --command="bash -c '${RUN_SCRIPT}'"

echo "Logs: sco acp jobs logs --workspace-name $WORKSPACE -j $jobname"