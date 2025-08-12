import pickle, os, re, random, torch
from swift.llm import get_model_tokenizer, load_dataset, get_template, LazyLLMDataset, RLHFArguments, rlhf_main
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import RLHFArgumentsMixin, RewardConfig, RLHFTrainerMixin
from datasets import Dataset
from functools import partial
import commons
import const_variable
import logging, warnings
class SuppressMultipleWarnings(logging.Filter):
    def filter(self, record):
        suppressed_msgs = [
            "Trainer.tokenizer is now deprecated",
            "System prompt modified, audio output may not work as expected"
        ]
        return not any(record.getMessage().startswith(msg) for msg in suppressed_msgs)
logging.getLogger().addFilter(SuppressMultipleWarnings())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger()
seed_everything(42)

# Hyperparameters for training
os.environ.update({
    "MAX_PIXELS": "1003520", # 250880 501760 1003520
    "NPROC_PER_NODE": "2",
    "ENABLE_AUDIO_OUTPUT": "0",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "PYTORCH_CUDA_ALLOC_CONF" : "expandable_segments:True"
})

data_seed = 42
###################################

cfg = RLHFArguments(
    output_dir='outputs/qwen25omni3b-think-balance-grspo-tokenintext',
    overwrite_output_dir=True,
    rlhf_type="grpo",
    model="Qwen/Qwen2.5-Omni-3B",
    reward_funcs=["external_r1v_acc", "format", "external_r1v_cosine", "repetition"],
    reward_weights=[2.0, 1.0, 1.0, 1.0],
    train_type="lora",
    lora_rank=4,
    lora_alpha=8,
    target_modules="all-linear",
    torch_dtype="bfloat16",
    use_hf=True,
    dataset=['arkiven4/cirdz-instruct'],
    split_dataset_ratio=0.02,
    system="/home/is/dwipraseetyo-a/NAS_HAI/Project/Qwen2.5-Omni/datas/grpo_system.txt",
    external_plugins=["/home/is/dwipraseetyo-a/NAS_HAI/Project/Qwen2.5-Omni/miscs/msswift_pluginsgrpo.py"],
    max_completion_length=2048,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    gpu_memory_utilization=0.95,
    # offload_optimizer=True,
    # offload_model=True,
    quant_method='bnb',
    quant_bits=4,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype='bfloat16',
    bnb_4bit_use_double_quant=True,
    attn_impl='flash_attention_2',
    #gradient_checkpointing=True,
    freeze_vit=True,
    gradient_accumulation_steps=1,
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    logging_steps=5,
    max_length=8192,
    warmup_ratio=0.05,
    dataloader_num_workers=16,
    dataset_num_proc=16,
    num_generations=2,
    temperature=1.0,
    top_p=0.99,
    top_k=50,
    #padding_side='left',
    label_names=['solution'],
    #device_map="none",
    deepspeed='zero2',
    log_entropy=True,
    top_entropy_quantile=0.2, #katanya sih more faster learning
    log_completions=True,
    # GSPO
    # importance_sampling_level='sequence', # deafault : token
    # epsilon=3e-4, # from paper section 5.1
    # epsilon_high=4e-4, # from paper section 5.1
    # # steps_per_generation=4, # from paper section 5.1 (each batch of rollout data is partitioned into four minibatches for gradient updates)
    # beta=0 # zero kl regularization https://github.com/volcengine/verl/pull/2775#issuecomment-3131807306
)

trainer = rlhf_main(args=cfg)
trainer.train()












# import commons
# import const_variable
# from my_datasets import QwenOmniFinetuneDataset



# import re
# from typing import Optional

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
#     matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
#     rewards = [1.0 if match else 0.0 for match in matches]
#     return rewards

# def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
#     """Reward function that checks if the completion matches the ground truth.
#     - If both gold and prediction are parseable → use math verification.
#     - If not parseable → compare as normalized text.
#     """
#     rewards = []
#     for completion, sol in zip(completions, solution):
#         reward = float(completion.strip().lower() == sol.strip().lower())
#         rewards.append(reward)
#     return rewards

# ##########################################################################################################
# commons.pretty_status("🧠 Loading Model...")

# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
#                         inference_mode=False, 
#                         r=8, 
#                         lora_alpha=32, 
#                         lora_dropout=0.05, 
#                         target_modules=["q_proj", "v_proj"])
#                         #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

# model, processor = get_OmniModel(model_path="Qwen/Qwen2.5-Omni-3B", processor_path="Qwen/Qwen2.5-Omni-3B", padding_side="left",
#                                 use_flash_attention=True, only_processor=False, quantize_4bit=True, 
#                                 offload_folder="offload", set_eval=False)

# ###
# # How about we finetuning the audio and image encoder, not using PEFT, or increase PEFT to audio and image encoder

# #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# #model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# #model = peft_model.unload()
# #del peft_model

# ##########################################################################################################


# #print(train_dataset[0])
# ##########################################################################################################
# commons.pretty_status("🚀 Start Training!")
# training_args = GRPOConfig(
#     output_dir="outputs/qwen25omni3b-think-balance-grpo",
#     learning_rate=1e-5,
#     remove_unused_columns=False,  # to access the solution column in accuracy_reward
#     num_train_epochs=1,
#     bf16=True,
#     # Parameters that control the data preprocessing
#     per_device_train_batch_size=2,
#     max_completion_length=1024,  # default: 256
#     num_generations=2,  # default: 8
#     max_prompt_length=2048,
#     # Parameters related to reporting and saving
#     report_to=["tensorboard"],
#     logging_steps=10,
#     push_to_hub=False,
#     save_strategy="steps",
#     save_steps=10,
# )
# training_args.remove_unused_columns = False

# trainer = GRPOTrainer(
#     model=model,
#     processing_class=processor,
#     reward_funcs=[format_reward, accuracy_reward],
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=dev_dataset,
#     #peft_config=peft_config,
# )

# trainer.train()
# #trainer.train(resume_from_checkpoint=f"{training_args.output_dir}/checkpoint-2000")
# trainer.save_model(training_args.output_dir)