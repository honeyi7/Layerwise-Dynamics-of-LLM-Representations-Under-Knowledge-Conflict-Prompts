#!/bin/bash
PARTITION=
WORKSPACE=
CONTAINER=
MOUNT=

jobname=causal_trace_qwen2.5_72B_parallel_$(date +%Y%m%d_%H%M%S)

NNODES=1
GPUS_PER_NODE=8

SCRIPT=main/causal_trace/code/causal_trace_main_72B.py

SPEC=N6lS.Iu.I80.${GPUS_PER_NODE}

MODEL_PATH=/path/to/Qwen2.5-72B-Instruct
KNOW_DATA=main/knowledge_probe/result/qwen2.5_72B/known_bucket.jsonl
UNKNOWN_DATA=main/knowledge_probe/result/qwen2.5_72B/unknown_bucket.jsonl
AMBIGUOUS_DATA=main/knowledge_probe/result/qwen2.5_72B/ambiguous_bucket.jsonl

INTERVENTION_KIND=attn  # attn mlp state
OUTPUT_DIR=main/causal_trace/results/qwen2.5_72B/${INTERVENTION_KIND}

MAX_SAMPLES=-1
BATCH_SIZE=20 
SAVE_INTERVAL=50

CONDA_ENV_PATH="/path/to/your/conda/envs"

PYTHON_CMD="${CONDA_ENV_PATH}/bin/python"
INSTALL_DEPS="${PYTHON_CMD} -m pip install accelerate --user"
MKDIR_CMD="mkdir -p ${OUTPUT_DIR}"


read -r -d '' RUN_SCRIPT << EOM
${ENV_SETUP}
export PATH=${CONDA_ENV_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:\$LD_LIBRARY_PATH
export PYTHONPATH=${CONDA_ENV_PATH}/lib/python3.13/site-packages:\$PYTHONPATH

${MKDIR_CMD}
${INSTALL_DEPS}

echo "Starting Parallel Processing on 8 GPUs..."

NUM_WORKERS=4
GPUS_PER_WORKER=2

pids=()

for (( i=0; i<\$NUM_WORKERS; i++ ))
do
    start_gpu=\$(( i * GPUS_PER_WORKER ))
    end_gpu=\$(( start_gpu + GPUS_PER_WORKER - 1 ))
    
    export CUDA_VISIBLE_DEVICES="\${start_gpu},\${end_gpu}"
    
    echo "Launching Worker \$i on GPUs \$CUDA_VISIBLE_DEVICES"
    
    nohup ${PYTHON_CMD} ${SCRIPT} \\
      --step run \\
      --model_path ${MODEL_PATH} \\
      --know_data_path ${KNOW_DATA} \\
      --unknown_data_path ${UNKNOWN_DATA} \\
      --ambiguous_data_path ${AMBIGUOUS_DATA} \\
      --output_dir ${OUTPUT_DIR} \\
      --intervention_kind ${INTERVENTION_KIND} \\
      --max_samples_per_dataset ${MAX_SAMPLES} \\
      --batch_size ${BATCH_SIZE} \\
      --save_interval ${SAVE_INTERVAL} \\
      --worker_id \$i \\
      --num_workers \$NUM_WORKERS \\
      > ${OUTPUT_DIR}/run_worker_\${i}.log 2>&1 &
      
    pids+=(\$!)
done

echo "Waiting for workers to finish..."
for pid in \${pids[*]}; do
    wait \$pid
done

echo "All workers finished. Starting Merge and Plot..."

export CUDA_VISIBLE_DEVICES=0
${PYTHON_CMD} ${SCRIPT} \\
  --step merge_and_plot \\
  --output_dir ${OUTPUT_DIR} \\
  --intervention_kind ${INTERVENTION_KIND}

echo "Job Completed Successfully."
EOM

# ---------------------------------------------------------

echo "======================================"
echo "Job: ${jobname}"
echo "Strategy: 4 Parallel Workers x 2 GPUs"
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
  --command="bash -c '${RUN_SCRIPT}'"

echo ""
echo "Submitted: ${jobname}"
echo "Check logs: sco acp jobs logs --workspace-name $WORKSPACE -j $jobname"