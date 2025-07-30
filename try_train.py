import pickle, os
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import librosa

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from qwen_omni_utils import process_mm_info

from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from datetime import datetime
from trl import SFTTrainer, SFTConfig

import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="System prompt modified, audio output may not work as expected. Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'")

import logging

class SuppressQwenWarning(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("System prompt modified, audio output may not work as expected")

# Add the filter to root logger or a specific one
logging.getLogger().addFilter(SuppressQwenWarning())

def pretty_status(message):
    line = "=" * (len(message) + 4)
    print(f"\n{line}")
    print(f"| {message} |")
    print(f"{line}\n")

def load_image_PIL(loaded_object):
    for obj in tqdm(loaded_object):
        for message in obj.get("messages", []):
            for content in message.get("content", []):
                if isinstance(content, dict) and "image" in content:
                    array = np.load(content["image"])
                    array_min = array.min()
                    array_max = array.max()
                    if array_max != array_min:
                        array_norm = (array - array_min) / (array_max - array_min)
                    else:
                        array_norm = np.zeros_like(array)
                    array_uint8 = (array_norm * 255).astype(np.uint8)
                    mask = array_uint8 != 253
                    coords = np.argwhere(mask)
                    if coords.size == 0:
                        cropped_array = array_uint8
                    else:
                        y0, x0 = coords.min(axis=0)
                        y1, x1 = coords.max(axis=0) + 1
                        cropped_array = array_uint8[y0:y1, x0:x1]
                    array_PIL = Image.fromarray(cropped_array)
                    content["image"] = array_PIL
    return loaded_object

class QwenOmniFinetuneDataset(Dataset):
    def __init__(self, data, processor, use_audio_in_video=False):
        self.data = data
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[idx]["messages"]
        return conversation

def collate_fn(conversations):
    text =  [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    batch = processor(text=text, audio=audios, images=images, videos=videos, 
                            return_tensors="pt", padding=True, use_audio_in_video=False)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Ignore the image token index in the loss computation (model specific)
    image_tokens = [151652, 151653, 151655] # image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    audio_tokens = [151647, 151648, 151646]

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    for audio_token_id in audio_tokens:
        labels[labels == audio_token_id] = -100  # Mask image token IDs in labels

    #inputs['use_audio_in_video']
    batch["labels"] = labels
    return batch

##########################################################################################################
pretty_status("🧠 Loading Model...")

# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
#                         inference_mode=False, 
#                         r=128, 
#                         lora_alpha=256, 
#                         lora_dropout=0, 
#                         target_modules=["q_proj", "k_proj", 
#                         "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                        inference_mode=False, 
                        r=8, 
                        lora_alpha=16, 
                        lora_dropout=0.05, 
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=None,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model) 
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

##########################################################################################################
pretty_status("📦 Loading Dataset...")
with open('/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/instruct.pkl.train', 'rb') as f:
    train_instruct = load_image_PIL(pickle.load(f))

with open('/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/instruct.pkl.dev', 'rb') as f:
    dev_instruct = load_image_PIL(pickle.load(f))

with open('/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/instruct.pkl.test', 'rb') as f:
    test_instruct = load_image_PIL(pickle.load(f))

train_dataset = QwenOmniFinetuneDataset(train_instruct, processor, use_audio_in_video=False)
dev_dataset = QwenOmniFinetuneDataset(dev_instruct, processor, use_audio_in_video=False)
test_dataset = QwenOmniFinetuneDataset(test_instruct, processor, use_audio_in_video=False)

print(train_dataset[0])

##########################################################################################################
pretty_status("🚀 Start Training!")
training_args = SFTConfig(
    output_dir="outputs/qwen25omni-7b-instructMedic-trl-sft",  # Directory to save the model
    logging_dir='outputs/qwen25omni-7b-instructMedic-trl-sft/logs',
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=4,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=40,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=40,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.05,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to=["tensorboard"],  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    # max_seq_length=1024  # Maximum sequence length for input
    remove_unused_columns=False,  # needed for multimodal input
)
training_args.remove_unused_columns = False

# training_args = TrainingArguments(
#     output_dir="./outputs/try1",
#     num_train_epochs=3,
#     per_device_train_batch_size = 1,
#     per_device_eval_batch_size = 1,
#     gradient_accumulation_steps = 4,
#     warmup_ratio = 0.1,
#     logging_dir='./outputs/try1/logs',
#     learning_rate = 1e-5,
#     logging_steps = 1,
#     eval_steps=75,
#     save_strategy="steps",
#     save_steps=100,
#     max_grad_norm=10.0,
#     fp16 = not torch.cuda.is_bf16_supported(),
#     gradient_checkpointing=True,
#     bf16 = torch.cuda.is_bf16_supported(),
#     optim = "adamw_8bit",
#     weight_decay = 0.001,
#     #seed = 3407,
#     lr_scheduler_type = "cosine",
#     remove_unused_columns=False,  # needed for multimodal input
#     #load_best_model_at_end=True,
#     report_to=["tensorboard"],
#     run_name=f"{datetime.now().strftime('%m-%d-%H-%M')}"
# )

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=collate_fn,
    #peft_config=peft_config,
    processing_class=processor.tokenizer,
)

trainer.train()
trainer.save_model(training_args.output_dir)