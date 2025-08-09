import pickle, os
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from qwen_omni_utils import process_mm_info
from trl import SFTTrainer, SFTConfig
import commons

import logging, warnings
class SuppressQwenWarning(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("System prompt modified, audio output may not work as expected")
logging.getLogger().addFilter(SuppressQwenWarning())
warnings.filterwarnings("ignore", message="System prompt modified, audio output may not work as expected. Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

commons.pretty_status("🕑 Evaluating Models...")

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
    image_tokens = [151652, 151653, 151655]
    audio_tokens = [151647, 151648, 151646]

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    for audio_token_id in audio_tokens:
        labels[labels == audio_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels
    return batch

##########################################################################################################
commons.pretty_status("🧠 Loading Model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=None,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    offload_folder="offload", # Offload weights to CPU if needed
)
#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
del peft_model

##########################################################################################################