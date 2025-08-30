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
from datasets import Dataset
from swift.llm import PtEngine, RequestConfig, InferRequest
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import commons
import const_variable

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def get_labelAnswer(text):
    sol_match = re.search(r'<answer>(.*?)</answer>', text, flags=re.IGNORECASE)
    answer = sol_match.group(1).strip() if sol_match else text.strip()
    label = disease_map.get(answer, 4)
    return label

disease_map = {
    "Healthy": 0,
    "Tuberculosis": 1,
    "Covid-19": 2,
    "Pneumonia": 3,
    "Others": 4,
}
disease_map_reverse = {v: k for k, v in disease_map.items()}

prompt_templates = {
    ("audio",): [
        "Based on the provided cough audio",
        "Listen to this cough sound",
        "Analyze the following cough sound"
    ],
    ("xray",): [
        "Examine this chest x-ray",
        "Does the provided x-ray image show",
        "Analyze the radiographic scan",
        "From this x-ray image"
    ],
    ("symptoms",): [
        "Given these symptoms",
        "Analyze the following symptoms",
        "Based on the symptom description"
    ],
    ("audio", "xray"): [
        "Based on the chest x-ray and cough audio",
        "Examine the x-ray and cough sound",
        "Given the radiograph and cough recording",
        "Using both the x-ray and cough audio"
    ],
    ("audio", "symptoms"): [
        "Given the cough audio and symptoms",
        "Analyze the patient symptoms and cough",
        "Using both the cough sound and symptoms"
    ],
    ("xray", "symptoms"): [
        "From the x-ray and symptoms",
        "Analyze the x-ray and clinical symptoms",
        "Given the chest scan and symptoms",
        "What is the diagnosis based on the x-ray and symptoms"
    ],
    ("audio", "xray", "symptoms"): [
        "Based on the x-ray, cough audio, and symptoms",
        "Evaluate the x-ray, cough sound, and symptoms",
        "Given the combined evidence",
        "Considering all inputs (image, sound, and symptoms)"
    ]
}

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['MAX_PIXELS'] = '250880'
os.environ['ENABLE_AUDIO_OUTPUT'] = '0'
os.environ['HF_HOME'] = '/home/is/dwipraseetyo-a/NAS_HAI/.cache'
DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz"

batch_size = 6
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
df = df.rename(columns={'coughdur': 'cough_duration', 'ngtsweats': 'night_sweets', 'weightloss': 'weight_loss', 'body_wt': 'body_weight'})

for modalities in list(prompt_templates.keys()):
    instruct_array_test = []
    for now_row in tqdm(df.itertuples(), desc=f"Processing Datasets", total=len(df)):
        row_dict = now_row._asdict()
        now_audiopath = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/" +  now_row.path_file_audio
        now_imgaepath = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/cidrz/" +  now_row.path_file_image

        #modalities = ["audio", "xray", "symptoms"]
        key = modalities #tuple(modalities)
        if key in prompt_templates:
            last_sentence_question = random.choice(prompt_templates[key]) + ", Determine what disease this could indicate, or answer 'Others' if you cannot determine."
        
        if now_row.path_file_audio == 'Unknown' and "audio" in modalities:
            continue

        answer = f"<answer>{disease_map_reverse[now_row.ground_truth_tb]}</answer>"
        question = ""
        modalities_tags = ""
        array_df = [None, None]
        if "symptoms" in modalities:
            row_dict = now_row._asdict()
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

    for temperature in [0.2]:
        adapter = f'{Checkpoint_PATH}/{Checkpoint_Number}'
        engine = PtEngine(model, adapters=adapter, max_batch_size=batch_size, use_hf=True)
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

        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
        plt.savefig(f"{Checkpoint_PATH}/cm_{temperature}.png")  # save image
        plt.close(fig)  # close to free memory

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0
            specificity = tn / (tn + fp) if (tn + fp) else 0
        else:  
            # calculate per-class sensitivity and specificity
            sensitivity = []
            specificity = []
            for i in range(len(cm)):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                sensitivity.append(tp / (tp + fn) if (tp + fn) else 0)
                specificity.append(tn / (tn + fp) if (tn + fp) else 0)

        with open(f"{Checkpoint_PATH}/result_{temperature}.txt", "a") as f:
            f.write(f"Modalities: {modalities}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            # If multi-class, sensitivity & specificity are lists
            if isinstance(sensitivity, list):
                f.write(f"Sensitivity: {', '.join(f'{s:.4f}' for s in sensitivity)}\n")
                f.write(f"Specificity: {', '.join(f'{s:.4f}' for s in specificity)}\n")
            else:
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