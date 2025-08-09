import pickle, os, re, random, torch
from peft import get_peft_model, LoraConfig, TaskType
from qwen_omni_utils import process_mm_info
from trl import SFTTrainer, SFTConfig
from qwen_mywrapper import get_OmniModel
from sentence_transformers import SentenceTransformer, util

import commons
import const_variable
from my_datasets import QwenOmniFinetuneDataset

import logging, warnings
class SuppressMultipleWarnings(logging.Filter):
    def filter(self, record):
        suppressed_msgs = [
            "Trainer.tokenizer is now deprecated",
            "System prompt modified, audio output may not work as expected"
        ]
        return not any(record.getMessage().startswith(msg) for msg in suppressed_msgs)
logging.getLogger().addFilter(SuppressMultipleWarnings())

# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message="System prompt modified, audio output may not work as expected. Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'")
# warnings.filterwarnings("ignore", message=r"Trainer\.tokenizer.*deprecated")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

embedder = SentenceTransformer('all-MiniLM-L6-v2')
pos_embeds = embedder.encode(const_variable.positive_templates, convert_to_tensor=True)
neg_embeds = embedder.encode(const_variable.negative_templates, convert_to_tensor=True)

def get_label_fromprompt(text, threshold=0.7):
    match = re.search(r"## 🧠 Overview\s*(.*?)\s*(##|$)", text, re.DOTALL)
    sentence = match.group(1).strip() if match else None

    if sentence is None:
        return random.randint(2, 4)
    
    # Compute embedding for sentence
    sent_embed = embedder.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities
    pos_sim = util.cos_sim(sent_embed, pos_embeds)  # shape: (1, N_pos)
    neg_sim = util.cos_sim(sent_embed, neg_embeds)  # shape: (1, N_neg)

    mean_pos_sim = pos_sim.mean().item()
    mean_neg_sim = neg_sim.mean().item()
    
    if mean_pos_sim > mean_neg_sim: # mean_pos_sim >= threshold and 
        return 1
    elif mean_neg_sim > mean_pos_sim:
        return 0
    else:
        return random.randint(2, 4)

def collate_fn(conversations):
    tb_labels = [get_label_fromprompt(conversation[-1]['content'][0]['text']) for conversation in conversations]
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
    batch["tb_labels"] = tb_labels
    return batch

##########################################################################################################
commons.pretty_status("🧠 Loading Model...")

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                        inference_mode=False, 
                        r=3, 
                        lora_alpha=6, 
                        lora_dropout=0.05, 
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

model, processor = get_OmniModel(model_path="Qwen/Qwen2.5-Omni-3B", processor_path="Qwen/Qwen2.5-Omni-3B", 
                                use_flash_attention=True, only_processor=False, quantize_4bit=True, 
                                offload_folder="offload", set_eval=False)

###
# How about we finetuning the audio and image encoder, not using PEFT, or increase PEFT to audio and image encoder

#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
# peft.get_peft_model_state_dict(model) -> Check affect vision text audio and thinker, janagn2 cuman thinker aj
model = peft_model.unload()
del peft_model

##########################################################################################################
commons.pretty_status("📦 Loading Dataset...")
with open('datas/instruct.pkl.train', 'rb') as f:
    train_instruct = commons.load_image_PIL(pickle.load(f))

with open('datas/instruct.pkl.dev', 'rb') as f:
    dev_instruct = commons.load_image_PIL(pickle.load(f))

train_dataset = QwenOmniFinetuneDataset(train_instruct, processor, use_audio_in_video=False)
dev_dataset = QwenOmniFinetuneDataset(dev_instruct, processor, use_audio_in_video=False)
print(train_dataset[0])

##########################################################################################################
commons.pretty_status("🚀 Start Training!")
training_args = SFTConfig(
    output_dir="outputs/qwen25omni-3b-instructMedic-reasonllm-notallpresent-trl-sft-sentencetrans",  # Directory to save the model
    logging_dir='outputs/qwen25omni-3b-instructMedic-reasonllm-notallpresent-trl-sft-sentencetrans/logs',
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=16,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=500,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=500,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=False,  # Load the best model after training
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
    label_names=["labels"],
    # max_seq_length=768,  # Maximum sequence length for input
    remove_unused_columns=False,  # needed for multimodal input
)
training_args.remove_unused_columns = False

class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the 
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100

            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        acc = 0.0
        sensitivity = 0.0
        if "tb_labels" in inputs:
            generated_texts = self.processing_class.batch_decode(predictions, skip_special_tokens=True)
            pred_labels = [get_label_fromprompt(text) for text in generated_texts]

            pred_tensor = torch.tensor(pred_labels, device=self.accelerator.device)
            true_tensor = torch.tensor(inputs["tb_labels"], device=self.accelerator.device)

            pred_tensor = self.accelerator.gather_for_metrics(pred_tensor)
            true_tensor = self.accelerator.gather_for_metrics(true_tensor)

            TP = ((pred_tensor == 1) & (true_tensor == 1)).sum().item()
            TN = ((pred_tensor == 0) & (true_tensor == 0)).sum().item()
            FP = ((pred_tensor == 1) & (true_tensor == 0)).sum().item()
            FN = ((pred_tensor == 0) & (true_tensor == 1)).sum().item()

            acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            self._metrics[mode]["sentence_accuracy"].append(acc)
            self._metrics[mode]["sentence_sensitivity"].append(sensitivity)
        
        if mode == "train":
            acc_penalty = (1.0 - acc)
            sens_penalty = (1.0 - sensitivity)
            loss = loss + (1.0 * acc_penalty) + (1.0 * sens_penalty)
        return (loss, outputs) if return_outputs else loss


trainer = CustomSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor,
)

trainer.train()
#trainer.train(resume_from_checkpoint=f"{training_args.output_dir}/checkpoint-2000")
trainer.save_model(training_args.output_dir)