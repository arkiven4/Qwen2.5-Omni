
#CUDA_HOME=/opt/nvidia/cuda-11.6u1 \
#uv run swift rlhf \
NCCL_BLOCKING_WAIT=1 \
NCCL_TIMEOUT=3600 \
OMP_NUM_THREADS=24 \
MAX_PIXELS=262144 \
NPROC_PER_NODE=2 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=0,1 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Omni-3B \
    --resume_only_model true \
    --ignore_data_skip true \
    --resume_from_checkpoint /home/is/dwipraseetyo-a/NAS_HAI/Project/Qwen2.5-Omni/outputs/qwen25omni3b-think-balance-grspo-tokenintext/v3-20250812-135107/checkpoint-1943 \
    --reward_funcs external_r1v_acc format external_r1v_cosine repetition \
    --reward_weights 2.0 1.0 1.0 0.5 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset arkiven4/cirdz-instruct \
    --use_hf true \
    --split_dataset_ratio 0.02 \
    --system datas/grpo_system.txt \
    --external_plugins miscs/msswift_pluginsgrpo.py \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --freeze_vit false \
    --freeze_llm true \
    --freeze_aligner false \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_use_double_quant true \
    --attn_impl flash_attn \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --num_generations 3 \
    --temperature 1.0 \
    --top_p 0.99 \
    --top_k 50 \
    --label_names solution \
    --log_entropy true \
    --top_entropy_quantile 0.2 \
    --log_completions true \
    --deepspeed zero2 \
    --importance_sampling_level sequence \
    --epsilon 0.0003 \
    --epsilon_high 0.0004 \
    --beta 0 \
    --output_dir outputs/qwen25omni3b-think-balance-grspo-tokenintext-phase2 \
    --overwrite_output_dir true