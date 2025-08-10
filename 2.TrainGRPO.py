import pickle, os, re, random, torch
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
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
    "MAX_PIXELS": "1003520",
    "NPROC_PER_NODE": "4",
    "ENABLE_AUDIO_OUTPUT": "0",
    "CUDA_VISIBLE_DEVICES": "0,1"
})

model_id_or_path = 'Qwen/Qwen2.5-Omni-3B'
output_dir = 'outputs/qwen25omni3b-think-balance-grpo'

data_seed = 42
max_length = 2048
split_dataset_ratio = 0.01  # Split validation set
num_proc = 4  # The number of processes for data loading.

# lora
lora_rank = 8
lora_alpha = 32

# training_args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['tensorboard'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')

# Obtain the model and template, and add a trainable Lora layer on the model.
model, tokenizer = get_model_tokenizer(model_id_or_path)
template = get_template(model.model_meta.template, tokenizer, default_system=const_variable.system_prompt, max_length=max_length)
template.set_mode('train')

target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# Print model structure and trainable parameters.
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

# Download and load the dataset, split it into a training set and a validation set,
# and encode the text data into tokens.
commons.pretty_status("📦 Loading Dataset...")
with open('datas/instruct_grpo_balance.pkl.dev', 'rb') as f:
    train_instruct = commons.load_image_PIL(pickle.load(f)[0:10])

with open('datas/instruct_grpo_balance.pkl.dev', 'rb') as f:
    dev_instruct = commons.load_image_PIL(pickle.load(f)[0:10])

train_dataset = Dataset.from_list(train_instruct) #commons.grpo_build_datasets(train_instruct, tokenizer)
val_dataset = Dataset.from_list(train_instruct)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# Print a sample
template.print_inputs(train_dataset[0])

# # Get the trainer and start the training.
# model.enable_input_require_grads()  # Compatible with gradient checkpointing
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     data_collator=template.data_collator,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     template=template,
# )
# trainer.train()

# last_model_checkpoint = trainer.state.last_model_checkpoint
# logger.info(f'last_model_checkpoint: {last_model_checkpoint}')














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