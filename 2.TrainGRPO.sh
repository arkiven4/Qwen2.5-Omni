NCCL_TIMEOUT=5400 \
RANK=0 \
WORLD_SIZE=1 \
CUDA_HOME=/home/is/dwipraseetyo-a/nvidia/cuda-12.8/ \
OMP_NUM_THREADS=24 \
MAX_PIXELS=250880 \
NPROC_PER_NODE=1 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=3 \
HF_HOME=/home/is/dwipraseetyo-a/NAS_HAI/.cache \
yes | swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Omni-3B \
    --reward_funcs external_r1v_acc external_r1v_cosine format repetition \
    --reward_weights 1.5 1.0 0.5 0.5 \
    --train_type lora \
    --lora_rank 3 \
    --lora_alpha 6 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset arkiven4/grpo_3modalities_datasets \
    --use_hf true \
    --system datas/grpo_system.txt \
    --external_plugins miscs/msswift_pluginsgrpo.py \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_use_double_quant true \
    --attn_impl 'flash_attention_2' \
    --deepspeed 'zero2' \
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
    --log_entropy true \
    --top_entropy_quantile 0.2 \
    --log_completions true \
    --importance_sampling_level sequence \
    --epsilon 0.0003 \
    --epsilon_high 0.0004 \
    --beta 0 \
    --output_dir outputs/qwen25omni3b-think-bigdata-grspo-tokenintext-250880-llmaligner \
    --overwrite_output_dir true