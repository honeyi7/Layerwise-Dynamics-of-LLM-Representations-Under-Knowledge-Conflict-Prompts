PARTITION=
WORKSPACE=
CONTAINER=
MOUNT=


jobname=probe_qwen25_72B_$(date +%Y%m%d_%H%M%S)


NNODES=1
GPUS_PER_NODE=8


SCRIPT=main/knowledge_probe/code/probe_main.py


SPEC=N6lS.Iu.I80.${GPUS_PER_NODE}


MODEL_PATH=/path/to/Qwen2.5-72B-Instruct
INPUT_FILE=counterfact/all.jsonl
OUTPUT_DIR=main/knowledge_probe/result/qwen2.5_72B


MKDIR_CMD="mkdir -p ${OUTPUT_DIR}"


MAIN_CMD="python3 ${SCRIPT} \
  --model_path ${MODEL_PATH} \
  --input_file ${INPUT_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --f1_threshold 0.5 \
  --tensor_parallel_size 8 \
  --max_new_tokens 32"


FULL_COMMAND="${MKDIR_CMD} && ${MAIN_CMD}"



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


echo "sco acp jobs logs --workspace-name $WORKSPACE -j $jobname"