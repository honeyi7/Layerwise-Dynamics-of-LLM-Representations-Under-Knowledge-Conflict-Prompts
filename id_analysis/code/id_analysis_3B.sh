
MODEL_SIZE="3B"


PARTITION=
WORKSPACE=
CONTAINER=
MOUNT=

jobname=id_analysis_qwen${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)


NNODES=1
GPUS_PER_NODE=1
SPEC=N6lS.Iu.I80.${GPUS_PER_NODE}


SCRIPT=main/id_analysis/code/id_analysis_main_single_gpu.py


MODEL_PATH="/path/to/Qwen2.5-${MODEL_SIZE}-Instruct"
PROBE_BASE="main/knowledge_probe/result/qwen2.5_${MODEL_SIZE}"
KNOW_DATA="${PROBE_BASE}/known_bucket.jsonl"
UNKNOWN_DATA="${PROBE_BASE}/unknown_bucket.jsonl"
AMBIGUOUS_DATA="${PROBE_BASE}/ambiguous_bucket.jsonl"

OUTPUT_DIR="main/id_analysis/results/qwen2.5_${MODEL_SIZE}"


BATCH_SIZE=50
MAX_SAMPLES=2000

CONDA_ENV_PATH="/path/to/your/conda/envs"
PYTHON_CMD="${CONDA_ENV_PATH}/bin/python"

read -r -d '' RUN_SCRIPT << EOM
${ENV_SETUP}
export PATH=${CONDA_ENV_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:\$LD_LIBRARY_PATH

mkdir -p ${OUTPUT_DIR}
${PYTHON_CMD} -m pip install scikit-learn scipy seaborn matplotlib accelerate --user

echo "Starting ID Analysis for Qwen2.5-${MODEL_SIZE}..."

# Step 1: Extraction (Single Process)
echo "Running Extraction..."
${PYTHON_CMD} ${SCRIPT} \\
  --step extract \\
  --model_path ${MODEL_PATH} \\
  --model_alias "Qwen2.5-${MODEL_SIZE}" \\
  --know_data_path ${KNOW_DATA} \\
  --unknown_data_path ${UNKNOWN_DATA} \\
  --ambiguous_data_path ${AMBIGUOUS_DATA} \\
  --output_dir ${OUTPUT_DIR} \\
  --max_samples ${MAX_SAMPLES} \\
  --batch_size ${BATCH_SIZE}

# Step 2: Analysis
echo "Running Analysis..."
${PYTHON_CMD} ${SCRIPT} \\
  --step analyze \\
  --output_dir ${OUTPUT_DIR} \\
  --model_alias "Qwen2.5-${MODEL_SIZE}"

echo "Job Completed. Output at: ${OUTPUT_DIR}"
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