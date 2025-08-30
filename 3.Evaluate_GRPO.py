import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['MAX_PIXELS'] = '752640'
os.environ['ENABLE_AUDIO_OUTPUT'] = '0'
DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz"

batch_size = 1
model = "Qwen/Qwen2.5-Omni-7B"
Checkpoint_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Project/ms-swift/outputs/250880-bigdata-sft-llmalign/v0-20250825-130255/"
Checkpoint_Number = "checkpoint-3587"

system_prompt = (
    "A conversation between User and Advanced medical assistant specialized in analyzing and diagnosing clinical conditions. and the Assistant determines whether the case is Positive or Negative. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

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
        answer = commons.generate_tb_response(modalities, now_row.llm_analyze_symptoms, now_row.llm_analyze_image, positive=(now_row.ground_truth_tb == 1), grpo=True)

        if now_row.path_file_audio == 'Unknown' and "audio" in modalities:
            continue
        
        question = ""
        modalities_tags = ""
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
            modalities_tags += "<audio>"
            array_df[0] = now_audiopath

        if "xray" in modalities:
            modalities_tags += "<image>"
            array_df[1] = now_imgaepath
            # xray_descriptions = ", ".join(
            #     f"{feat.replace('_', ' ')} is {row_dict[feat]}"
            #     for feat in const_variable.columns_imagefeat
            #     if row_dict.get(feat) != "Unknown"
            # )
            # if xray_descriptions:
            #     question += f" The chest x-ray metadata are {xray_descriptions}."

        question = question.strip()
        question = question.rstrip(",.")
        if not question.endswith("."):
            question += "."

        question = modalities_tags + "" + question
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
            temp_instruct["messages"][1]['content'].append({"type": "image", "image":array_df[1]})
        instruct_array_test.append(temp_instruct)

    test_instruct = commons.load_image_PIL(instruct_array_test)
    test_dataset = Dataset.from_list(commons.grpo_build_datasets(test_instruct, None))

    for temperature in [0.0, 0.5, 1.0]:
        adapter = f'{Checkpoint_PATH}/{Checkpoint_Number}'
        engine = PtEngine(model, adapters=adapter, max_batch_size=batch_size)
        request_config = RequestConfig(max_tokens=512, temperature=temperature) # Kenapa 0, padahal di config 1
        
        resp_list = engine.infer(list(test_dataset), request_config)
        y_true = []
        y_pred= []
        for idx, resp_now in enumerate(resp_list):
            model_response = resp_now.choices[0].message.content
            y_pred.append(get_labelAnswer(model_response))
            y_true.append(get_labelAnswer(test_dataset[idx]['solution']))

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0

        with open(f"{Checkpoint_PATH}/results_501760_{temperature}.txt", "a") as f:
            f.write(f"Modalities: {modalities}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Sensitivity: {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            f.write("-" * 40 + "\n")  # separator between runs

        del engine, request_config, resp_list
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()