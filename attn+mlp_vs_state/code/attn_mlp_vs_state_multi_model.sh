
source path/to/your/conda/activate


conda activate path/to/your/conda/envs

echo "Environment activated: $(which python)"


MODELS=(
    "qwen2.5_0.5B"
    "qwen2.5_1.5B"
    "qwen2.5_3B"
    "qwen2.5_7B"
    "qwen2.5_14B"
    "qwen2.5_32B"
    "qwen2.5_72B"
)

SCRIPT_PATH=main/attn+mlp_vs_state/code/attn+mlp_vs_state_multi_model.py

echo "=========================================="
echo "Starting Mechanism Verification Pipeline"
echo "=========================================="



for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Processing Model: $model"
    
    python "$SCRIPT_PATH" --model_name "$model"
    
    if [ $? -eq 0 ]; then
        echo ">>> Successfully processed $model"
    else
        echo ">>> [ERROR] Failed to process $model"
    fi
done

echo ""
echo "=========================================="
echo "All Done."
echo "=========================================="