import pickle, os, re, random, torch, json
from peft import get_peft_model, LoraConfig, TaskType, get_peft_model_state_dict
from qwen_omni_utils import process_mm_info
from trl import SFTTrainer, SFTConfig
from my_qwenwrapper import get_OmniModel
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

import commons
import const_variable
from my_datasets import QwenOmniFinetuneDataset

import re
def clean_text_for_log(text):
    # Replace any CR, LF, CRLF with literal \n
    text = re.sub(r'[\r\n]+', r'\\n', text)
    # Optionally replace tabs with space
    text = text.replace('\t', ' ')
    # Strip leading/trailing spaces
    text = text.strip()
    return text

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

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
# pos_embeds = embedder.encode(const_variable.positive_templates, convert_to_tensor=True)
# neg_embeds = embedder.encode(const_variable.negative_templates, convert_to_tensor=True)

def get_label_fromprompt(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    last_answer = matches[-1].strip() if matches else text.strip()
    return last_answer

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

    batch['use_audio_in_video'] = False
    batch["labels"] = labels
    batch["tb_labels"] = tb_labels
    return batch

##########################################################################################################
commons.pretty_status("🧠 Loading Model...")

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                        inference_mode=False, 
                        r=4, 
                        lora_alpha=8, 
                        lora_dropout=0.05, 
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

model, processor = get_OmniModel(model_path="Qwen/Qwen2.5-Omni-3B", processor_path="Qwen/Qwen2.5-Omni-3B", 
                                use_flash_attention=True, only_processor=False, quantize_4bit=False, 
                                offload_folder=None, set_eval=False)

###
# How about we finetuning the audio and image encoder, not using PEFT, or increase PEFT to audio and image encoder

#model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
#model.gradient_checkpointing_enable() # Not Work If Deepspeed
model.enable_input_require_grads()
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
model = peft_model.unload()
del peft_model

##########################################################################################################
commons.pretty_status("📦 Loading Dataset...")
with open('datas/instruct_sft_balance.pkl.train', 'rb') as f:
    train_instruct = commons.load_image_PIL(pickle.load(f))

with open('datas/instruct_sft_balance.pkl.dev', 'rb') as f:
    dev_instruct = commons.load_image_PIL(pickle.load(f))

train_dataset = QwenOmniFinetuneDataset(train_instruct, processor, use_audio_in_video=False)
dev_dataset = QwenOmniFinetuneDataset(dev_instruct, processor, use_audio_in_video=False)
print(train_dataset[0])

##########################################################################################################
commons.pretty_status("🚀 Start Training!")
training_args = SFTConfig(
    output_dir="outputs/qwen25omni3b-reason-notallpresent-trl-sft-balance",  # Directory to save the model
    logging_dir='outputs/qwen25omni3b-reason-notallpresent-trl-sft-balance/logs',
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency, Use Internal For Deepspeed
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=100,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=100,  # Steps interval for saving
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
            generated_texts = self.processing_class.batch_decode(predictions, skip_special_tokens=False)
            pred_labels = [get_label_fromprompt(text) for text in generated_texts]

            TP, TN, FP, FN = 0, 0, 0, 0
            for pred, truth in zip(pred_labels, inputs["tb_labels"]):
                pred_is_pos = "positive" in pred.lower().strip()
                truth_is_pos = "positive" in truth.lower().strip()

                if pred_is_pos and truth_is_pos:
                    TP += 1
                elif not pred_is_pos and not truth_is_pos:
                    TN += 1
                elif pred_is_pos and not truth_is_pos:
                    FP += 1
                elif not pred_is_pos and truth_is_pos:
                    FN += 1

            acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
            sensitivity = TP / (TP + FN) if (TP + FN) else 0
            if sensitivity < 0.3 or acc < 0.4:
                with open("generated_texts.log", "a", encoding="utf-8") as f:
                    for text, pred_label, ref_label in zip(generated_texts, pred_labels, inputs["tb_labels"]):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        clean_text = json.dumps(text.strip())
                        f.write(f"[LOG: {timestamp}] Generated: {clean_text} \n| Pred: {pred_label} | Ref: {ref_label}\n")
            self._metrics[mode]["sentence_accuracy"].append(acc)
            self._metrics[mode]["sentence_sensitivity"].append(sensitivity)
        
        if mode == "train":
            acc_penalty = (1.0 - acc)
            sens_penalty = (1.0 - sensitivity)
            loss = loss + (1.5 * acc_penalty) + (1.5 * sens_penalty)
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

#trainer.train()
trainer.train(resume_from_checkpoint=f"{training_args.output_dir}/checkpoint-333")
trainer.save_model(training_args.output_dir)