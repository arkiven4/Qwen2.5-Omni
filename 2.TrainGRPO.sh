OMP_NUM_THREADS=14 \
CUDA_HOME=/opt/nvidia/cuda-12.2u2 \
MAX_PIXELS=1003520 \
NPROC_PER_NODE=2 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=0,1 \
uv run swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Omni-3B \
    --reward_funcs external_r1v_acc format external_r1v_cosine repetition \
    --reward_weights 2.0 1.0 1.0 0.5 \
    --train_type lora \
    --lora_rank 4 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset arkiven4/cirdz-instruct \
    --use_hf True \
    --split_dataset_ratio 0.02 \
    --system /work/dwipraseetyo-a/Qwen2.5-Omni/datas/grpo_system.txt \
    --external_plugins /work/dwipraseetyo-a/Qwen2.5-Omni/miscs/msswift_pluginsgrpo.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --freeze_vit True \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --num_generations 2 \
    --temperature 1.0 \
    --top_p 0.99 \
    --top_k 50 \
    --label_names solution \
    --log_entropy True \
    --top_entropy_quantile 0.2 \
    --log_completions True \
    --deepspeed zero2 \
    --importance_sampling_level sequence \
    --epsilon 0.0003 \
    --epsilon_high 0.0004 \
    --beta 0 \
    --output_dir outputs/qwen25omni3b-think-balance-grspo-tokenintext \
    --overwrite_output_dir True
