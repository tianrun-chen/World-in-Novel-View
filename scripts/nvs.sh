#! /bin/bash
# 
# Usage
# 
# 2 npus Training (DDP)
# bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc geope --npus "0,1"
# 
# 2 npus Training (FSDP)
# bash ./scripts/nvs.sh --ray_encoding plucker --pos_enc geope --npus "0,1" --use-fsdp
# 

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ray_encoding)
      RAY_ENCODING="$2"   
      shift 2
      ;;
    --pos_enc)
      POS_ENC="$2"
      shift 2
      ;;
    --npus)
      npus="$2"
      shift 2
      ;;
    --test-zoom-in)
      TEST_ZOOM_IN="$2"
      shift 2
      ;;
    --test-context-views)
      TEST_CONTEXT_VIEWS="$2"
      shift 2
      ;;
    --test-render-video)
      TEST_RENDER_VIDEO=true
      shift 1
      ;;
    --use-fsdp)
      USE_FSDP=true
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 --ray_encoding <ray_encoding> --pos_enc <pos_enc> --npus <gpu_list> [options]"
      echo "  --ray_encoding: plucker, camray, none, or raymap"
      echo "  --pos_enc: geope, gta, or none"
      echo "  --npus: comma-separated GPU list (e.g., '0,1')"
      echo "  --use-fsdp: Use FSDP instead of DDP for training"
      echo "  --test-zoom-in: space-separated zoom factors for testing (e.g., '3 5')"
      echo "  --test-context-views: space-separated context views for testing (e.g., '2 4 8 16')"
      echo "  --test-render-video: render video for testing"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$RAY_ENCODING" ]; then
  echo "Error: --ray_encoding is required"
  exit 1
fi
if [ -z "$POS_ENC" ]; then
  echo "Error: --pos_enc is required"
  exit 1
fi
if [ -z "$npus" ]; then
  echo "Error: --npus is required"
  exit 1
fi
Nnpus=$(echo $npus | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES=$npus

NAME="release-${Nnpus}npus-b8-s1-80k"
TORCHRUN_CMD="NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=$Nnpus"

# --- STEP 1: REMOVE --output_dir FROM THIS ARRAY ---
BASE_ARGS=(
    "nvs/trainval_modi.py lvsm"
    "--amp" "--amp_dtype" "fp16"
    "--dataset_batch_scenes" "8"
    "--dataset_supervise_views" "1"
    "--model_config.encoder.num_layers" "24"
    "--model_config.encoder.layer.d_model" "768"
    "--model_config.encoder.layer.nhead" "16"
    "--model_config.encoder.layer.dim_feedforward" "3072"
    "--model_config.encoder.layer.qk_norm"
    "--auto_resume"
    "--max_steps" "2000000" "--test_every" "10000"
    "--test_index_fp" "./assets/evaluation_index_re10k_context2.json"
    "--model_config.ray_encoding" "${RAY_ENCODING}"
    "--model_config.pos_enc" "${POS_ENC}"
    # The --output_dir line that was here is now REMOVED.
)

# --- STEP 2: UPDATE NAME FOR FSDP (NO LONGER MODIFIES THE ARRAY) ---
if [ "$USE_FSDP" = true ]; then
    echo "FSDP mode enabled."
    BASE_ARGS+=(
        # "--use_fsdp" "True"
        "--fsdp_sharding_strategy" "FULL_SHARD"
        "--fsdp_mixed_precision" "fp16"
    )
    # Just update the NAME variable
    NAME="${NAME}-fsdp"
fi

# --- STEP 3: ADD THE CORRECT --output_dir AT THE END ---
BASE_ARGS+=(
    "--output_dir" "results/${NAME}-${RAY_ENCODING}-${POS_ENC}"
)


echo "NAME: ${NAME}"
echo "RAY_ENCODING: ${RAY_ENCODING}"
echo "POS_ENC: ${POS_ENC}"

if [ -n "$TEST_ZOOM_IN" ]; then
    for zoom_factor in $TEST_ZOOM_IN; do
        echo "Starting testing with zoom factor ${zoom_factor}..."
        TEST_ARGS=(
            "${BASE_ARGS[@]}"
            "--test_only --auto_resume"
            "--test_zoom_factor ${zoom_factor}"
            "--test_subdir eval-zoom${zoom_factor}x"
        )
        eval $TORCHRUN_CMD "${TEST_ARGS[@]}"
    done
    exit 0
elif [ -n "$TEST_CONTEXT_VIEWS" ]; then
    for context_views in $TEST_CONTEXT_VIEWS; do
        echo "Starting testing with ${context_views} context views..."
        TEST_ARGS=(
            "${BASE_ARGS[@]}"
            "--test_only --auto_resume"
            "--model_config.ref_views ${context_views}"
            "--test_input_views ${context_views}"
            "--test_index_fp ./assets/evaluation_index_re10k_context${context_views}.json"
            "--test_subdir eval-context${context_views}"
            "--render_video"
        )
        eval $TORCHRUN_CMD "${TEST_ARGS[@]}"
    done
    exit 0
elif [ -n "$TEST_RENDER_VIDEO" ]; then
    echo "Starting testing with rendering video for fisrt 10 scenes ..."
    TEST_ARGS=(
        "${BASE_ARGS[@]}"
        "--test_only --auto_resume --render_video --test_n 10"
    )
    eval $TORCHRUN_CMD "${TEST_ARGS[@]}"
    exit 0
else
    echo "Starting training process..."
    TRAIN_ARGS=(
        "${BASE_ARGS[@]}"
    )
    # (修改) 使用更标准的命令执行方式
    eval $TORCHRUN_CMD "${TRAIN_ARGS[@]}"
    exit 0
fi