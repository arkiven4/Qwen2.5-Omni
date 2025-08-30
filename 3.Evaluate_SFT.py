import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

import torch
device_name = torch.cuda.get_device_name(0)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"Logical index: {i}, Name: {props.name}")

import pickle, os, re, random
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from qwen_omni_utils import process_mm_info
from my_qwenwrapper import get_OmniModel, get_stoppingcrit
from datasets import Dataset
from swift.llm import PtEngine, RequestConfig, InferRequest
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import commons
import const_variable
from my_datasets import QwenOmniFinetuneDataset

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def get_labelAnswer(text):
    sol_match = re.search(r'<answer>(.*?)</answer>', text, flags=re.IGNORECASE)
    answer = sol_match.group(1).strip() if sol_match else text.strip()
    label = 1 if answer == "Positive Tuberculosis" else 0
    return label

def collateevaluate_fn(conversations):
    conversations_copy = copy.deepcopy(conversations)
    tb_labels  = []
    for conv in conversations_copy:
        tb_labels.append(get_labelAnswer(conv[2]['content'][0]['text']))
        del conv[2]
    
    text =  processor.apply_chat_template(conversations_copy, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations_copy, use_audio_in_video=False)
    batch = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
    
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
    batch['tb_labels'] = tb_labels
    return batch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz"

Checkpoint_PATH = "outputs/qwen25omni3b-noreason-notallpresent-trl-sft-sentencetrans"
Checkpoint_Number = ""

system_prompt = '''
You are an advanced medical assistant AI specialized in analyzing and diagnosing clinical conditions, capable of perceiving auditory and visual inputs. 
You can interpret and reason over various medical inputs, including auditory inputs, visual inputs, and patient symptoms, individually or in combination, depending on what is provided. 
Your task is to analyze the given input, explain your reasoning, and give a possible diagnosis. Answer are enclosed within <answer> </answer> tags, respectively, i.e., <answer> answer here </answer> .
Always respond in the following format:

## ⚠️ Points to Review and Disclaimer
<If no auditory or visual input is provided>

## 🧠 Overview
<answer> <Positive or Negative Diagnosis> </answer>

## 📋 Observations
**Chest X-ray:**
<Your explanation based on the relevant visual input>"

**Symptoms:**
<Your explanation based on the input symptoms>"

**Audio:**
<Your explanation based on the input audio>"
'''

model, processor = get_OmniModel(model_path="Qwen/Qwen2.5-Omni-3B", processor_path="Qwen/Qwen2.5-Omni-3B", 
                                 adapter_path=f"{Checkpoint_PATH}/{Checkpoint_Number}", 
                                 use_flash_attention=False, only_processor=False, quantize_4bit=True, offload_folder="offload", set_eval=True)
stopping_criteria = get_stoppingcrit(processor)

random.seed(42)
df = pd.read_csv(f"{DATA_PATH}/metadata_cut_processed.csv.test")
df_llm_symptoms = ( pd.read_csv(f"datas/reasoning/symptoms/gpt-4o-mini_symptoms.csv.test").groupby('barcode', group_keys=False).apply(lambda x: x.sample(1), include_groups=True).reset_index(drop=True) ) 
df_llm_images = ( pd.read_csv(f"datas/reasoning/xray/medgemma_xray_formatted.csv.test").groupby('path_file_image', group_keys=False).apply(lambda x: x.sample(1), include_groups=True).reset_index(drop=True) )
df = pd.merge(df, df_llm_symptoms, on='barcode', how='left')
df = pd.merge(df, df_llm_images, on='path_file_image', how='left')
df = df.rename(columns={'coughdur': 'cough_duration', 'ngtsweats': 'night_sweets', 'weightloss': 'weight_loss', 'body_wt': 'body_weight'})

for modalities in list(const_variable.prompt_templates.keys()):
    instruct_array_test = []
    for now_row in tqdm(df.itertuples(), desc=f"Processing Datasets", total=len(df)):
        row_dict = now_row._asdict()
        now_audiopath = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/" +  now_row.path_file_audio
        now_imgaepath = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/" +  now_row.path_file_image

        #modalities = ["audio", "xray", "symptoms"]
        key = modalities #tuple(modalities)
        if key in const_variable.prompt_templates:
            last_sentence_question = random.choice(const_variable.prompt_templates[key]) + ". "
        answer = commons.generate_tb_response(modalities, now_row.llm_analyze_symptoms, now_row.llm_analyze_image, positive=(now_row.ground_truth_tb == 1), grpo=False)

        if now_row.path_file_audio == 'Unknown' and "audio" in modalities:
            continue
        
        question = ""
        array_df = [None, None]
        if "symptoms" in modalities:
            row_dict = now_row._asdict()
            #selected_feats = random.sample(const_variable.columns_soundfeat, k=random.randint(3, len(const_variable.columns_soundfeat)))
            symptom_descriptions = ", ".join(
                f"{feat.replace('_', ' ')} is {row_dict[feat]}"
                for feat in const_variable.columns_soundfeat
                if row_dict.get(feat) != "Unknown"
            )
            if symptom_descriptions:
                question += f"The patient symptoms are {symptom_descriptions}."

        if "audio" in modalities:
            array_df[0] = now_audiopath

        if "xray" in modalities:
            array_df[1] = now_imgaepath
            xray_descriptions = ", ".join(
                f"{feat.replace('_', ' ')} is {row_dict[feat]}"
                for feat in const_variable.columns_imagefeat
                if row_dict.get(feat) != "Unknown"
            )
            if xray_descriptions:
                question += f" The chest x-ray metadata are {xray_descriptions}."

        question = question.strip()
        question = question.rstrip(",.")
        if not question.endswith("."):
            question += "."

        question += " " + last_sentence_question

        temp_instruct = {"messages": [
            {"role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [
                {"type": "text", "text": answer}]},
        ]}
        if array_df[0] != None:
            temp_instruct["messages"][1]['content'].append({"type": "audio", "audio": array_df[0]})
        if array_df[1] != None:
            temp_instruct["messages"][1]['content'].append({"type": "image", "image": array_df[1]})
        instruct_array_test.append(temp_instruct)

    test_instruct = commons.load_image_PIL(instruct_array_test)
    test_dataset = QwenOmniFinetuneDataset(test_instruct, processor, use_audio_in_video=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collateevaluate_fn)

    for temperature in [0.0, 0.5, 1.0]:
        y_true = []
        y_pred= []
        for batch in tqdm(test_loader):
            y_true.extend(batch['tb_labels'])
            del batch['tb_labels']
            batch = { k: (v.to(model.device).to(model.dtype) if v.dtype.is_floating_point else v.to(model.device))
                if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad(), torch.amp.autocast('cuda'):
                generation = model.generate(
                    **batch,
                    stopping_criteria=stopping_criteria,
                    max_new_tokens=768,
                )
                generate_ids = generation[:, batch["input_ids"].size(1):]

            decoded = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True, do_sample=True, top_p = 1.0, top_k = 50, temperature=temperature)
            y_pred.extend([get_labelAnswer(now_answer) for now_answer in decoded])
            del generation, generate_ids, batch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0

        with open(f"{Checkpoint_PATH}/results_{temperature}.txt", "a") as f:
            f.write(f"Modalities: {modalities}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write("-" * 40 + "\n")  # separator between runs

        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()