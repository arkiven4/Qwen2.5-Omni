import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
import os
import librosa
import random
import pickle
import pydicom
import requests
import torch
import re
import shutil
from pydicom.datadict import keyword_for_tag
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from IPython.display import Markdown, display
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from sklearn.utils import resample
import random
from itertools import combinations

import commons
import const_variable

from huggingface_hub import login
from dotenv import load_dotenv
from datasets import Dataset, concatenate_datasets

random.seed(42)
ALL_COMBO = [
    ("path_file_audio",),
    ("path_file_image",),
    ("symptoms_sentence",),
    ("path_file_audio", "path_file_image"),
    ("path_file_audio", "symptoms_sentence"),
    ("path_file_image", "symptoms_sentence"),
    ("path_file_audio", "path_file_image", "symptoms_sentence"),
]
mod_map = {
    "path_file_audio": "audio",
    "path_file_image": "xray",
    "symptoms_sentence": "symptoms"
}

def sample_modalities_numeric_overlap(df, seed=42, max_per_modality=21):
    random.seed(seed)
    np.random.seed(seed)

    for col in ['path_file_audio', 'path_file_image']:
        df[col] = df[col].apply(lambda x: os.path.join(
            'validation', os.path.basename(x)) if pd.notna(x) else np.nan)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[df['disease'].isin([1, 2, 0])].reset_index(drop=True)

    augmented_rows = []
    disease_classes = df['disease'].unique()
    for label in tqdm(disease_classes, desc="Processing disease classes"):
        class_df = df[df['disease'] == label].reset_index(drop=True)

        audio_symp_rows = class_df[~class_df['path_file_audio'].isna() | ~class_df['symptoms_sentence'].isna()]
        image_rows = class_df[~class_df['path_file_image'].isna()]

        combo_to_rows = {combo: [] for combo in ALL_COMBO}
        for _, row in audio_symp_rows.iterrows():
            base_audio = row['path_file_audio'] if pd.notna(row['path_file_audio']) else None
            base_symp = row['symptoms_sentence'] if pd.notna(row['symptoms_sentence']) else None
            base_image = image_rows.sample(1).iloc[0]['path_file_image']

            combos = {
                ("path_file_audio",): {"path_file_audio": base_audio, "path_file_image": None, "symptoms_sentence": None},
                ("path_file_image",): {"path_file_audio": None, "path_file_image": base_image, "symptoms_sentence": None},
                ("symptoms_sentence",): {"path_file_audio": None, "path_file_image": None, "symptoms_sentence": base_symp},
                ("path_file_audio", "path_file_image"): {"path_file_audio": base_audio, "path_file_image": base_image, "symptoms_sentence": None},
                ("path_file_audio", "symptoms_sentence"): {"path_file_audio": base_audio, "path_file_image": None, "symptoms_sentence": base_symp},
                ("path_file_image", "symptoms_sentence"): {"path_file_audio": None, "path_file_image": base_image, "symptoms_sentence": base_symp},
                ("path_file_audio", "path_file_image", "symptoms_sentence"): {"path_file_audio": base_audio, "path_file_image": base_image, "symptoms_sentence": base_symp},
            }

            for combo, values in combos.items():
                new_row = row.copy()
                for col, val in values.items():
                    new_row[col] = val
                new_row["modalities_simple"] = tuple(mod_map[m] for m in combo if m in mod_map)
                combo_to_rows[combo].append(new_row)

        for combo, rows in combo_to_rows.items():
            augmented_rows.extend(rows[:max_per_modality])

    final_df = pd.DataFrame(augmented_rows)
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return final_df

def pop_row(rows):
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    rows.drop(rows.index[0], inplace=True)
    return row

def sample_modalities_numeric(df, split, maximum_disease=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    for col in ['path_file_audio', 'path_file_image']:
        df[col] = df[col].apply(lambda x: os.path.join(split, os.path.basename(x)) if pd.notna(x) else np.nan)

    augmented_rows = []
    df = df[df['disease'].isin([0, 1, 2])].reset_index(drop=True)
    disease_classes = df['disease'].unique()
    min_len = maximum_disease if maximum_disease != None else df['disease'].value_counts().min()
    for label in tqdm(disease_classes, desc="Processing disease classes"):
        class_df = df[df['disease'] == label].reset_index(drop=True)

        audio_rows = class_df[~class_df['path_file_audio'].isna()].copy()
        image_rows = class_df[~class_df['path_file_image'].isna()].copy()
        symptoms_rows = class_df[~class_df['symptoms_sentence'].isna()].copy()

        audio_rows = audio_rows.sample(frac=1, random_state=42).reset_index(drop=True)
        image_rows = image_rows.sample(frac=1, random_state=42).reset_index(drop=True)
        symptoms_rows = symptoms_rows.sample(frac=1, random_state=42).reset_index(drop=True)

        combos = []
        while len(combos) <= min_len and (len(audio_rows) > 0 or len(image_rows) > 0 or len(symptoms_rows) > 0):
            for combo_keys in ALL_COMBO:
                row_dict = {"path_file_audio": None, "path_file_image": None, "symptoms_sentence": None}
                if "path_file_audio" in combo_keys:
                    base_audio = pop_row(audio_rows)
                    row_dict["path_file_audio"] = base_audio["path_file_audio"] if base_audio is not None else None
                if "path_file_image" in combo_keys:
                    base_image = pop_row(image_rows)
                    row_dict["path_file_image"] = base_image["path_file_image"] if base_image is not None else None
                    row_dict["llm_analyze_image"] = base_image["llm_analyze_image"] if base_image is not None else None
                if "symptoms_sentence" in combo_keys:
                    base_symp = pop_row(symptoms_rows)
                    row_dict["symptoms_sentence"] = base_symp["symptoms_sentence"] if base_symp is not None else None
                    row_dict["llm_analyze_symptoms"] = base_symp["llm_analyze_symptoms"] if base_symp is not None else None
                if any(row_dict.values()):
                    combos.append((combo_keys, row_dict))

        for combo, values in combos:
            row_push = {
                "modalities_simple": tuple(mod_map[m] for m in combo if m in mod_map),
                "disease": label
            }
            for col, val in values.items():
                row_push[col] = val

            augmented_rows.append(row_push)

    final_df = pd.DataFrame(augmented_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    # min_count = final_df.groupby(['disease', 'modalities_simple']).size().min()
    # df_balanced = (final_df.groupby(['disease', 'modalities_simple'], group_keys=False).apply(lambda x: resample(x, replace=False, n_samples=max(min_count, min_len), random_state=42)))
    # df_balanced.reset_index(drop=True, inplace=True)
    return final_df

disease_map = {
    "Others": 0,
    "Tuberculosis": 1,
    "Covid-19": 2
}
disease_map_reverse = {v: k for k, v in disease_map.items()}

prompt_templates = {
    ("audio",): [
        "Based on the cough audio provided,",
        "Evaluate the cough recording carefully,",
        "Consider the characteristics of this cough audio,",
        "From the recorded cough sound,",
        "Assess the features of the cough audio,",
        "Interpret the provided cough recording,",
        "Review the cough audio carefully,",
        "Examine the cough sound,"
    ],
    ("xray",): [
        "Based on the chest x-ray image,",
        "Evaluate the features seen in the x-ray,",
        "Interpret the chest radiograph,",
        "From the provided x-ray scan,",
        "Assess the chest x-ray image,",
        "Review the radiographic scan carefully,",
        "Examine the x-ray image,",
        "Consider the details in this chest x-ray,"
    ],
    ("symptoms",): [
        "Based on the reported symptoms,",
        "Evaluate the patient symptoms,",
        "Interpret the described symptom information,",
        "Review the presented symptoms,",
        "Assess the clinical symptoms carefully,",
        "Examine the provided symptom information,",
        "Consider the described symptoms,",
        "Analyze the symptom details,"
    ],
    ("audio", "xray"): [
        "Based on both the cough audio and chest x-ray,",
        "Evaluate the patient using cough audio and x-ray together,",
        "Interpret the cough recording along with the chest x-ray,",
        "Assess the x-ray and cough audio,",
        "Review the cough sound and chest x-ray together,",
        "Examine both the cough recording and x-ray image,",
        "Consider the x-ray alongside the cough audio,",
        "Analyze the combined cough audio and chest x-ray,"
    ],
    ("audio", "symptoms"): [
        "Based on the cough audio and reported symptoms,",
        "Evaluate the patient using cough audio and symptoms together,",
        "Interpret the cough recording along with the patient symptoms,",
        "Assess the cough sound and symptom description,",
        "Review the combined cough audio and symptoms,",
        "Examine both the cough recording and symptoms,",
        "Consider the cough audio alongside symptom information,",
        "Analyze the cough audio with the reported symptoms,"
    ],
    ("xray", "symptoms"): [
        "Based on the chest x-ray and reported symptoms,",
        "Evaluate the patient using x-ray and symptoms together,",
        "Interpret the chest x-ray along with the patient symptoms,",
        "Assess the x-ray and symptom details,",
        "Review the combined x-ray and symptom information,",
        "Examine both the chest radiograph and symptoms,",
        "Consider the x-ray image alongside the reported symptoms,",
        "Analyze the x-ray together with the symptom description,"
    ],
    ("audio", "xray", "symptoms"): [
        "Based on the cough audio, chest x-ray, and symptoms,",
        "Evaluate the patient using audio, x-ray, and symptoms together,",
        "Interpret the combined cough recording, x-ray, and symptom information,",
        "Assess all inputs: cough audio, x-ray, and symptoms,",
        "Review the cough sound, chest x-ray, and symptom details,",
        "Examine the cough audio, x-ray scan, and patient symptoms,",
        "Consider all data from audio, x-ray, and symptoms,",
        "Analyze the cough recording, x-ray image, and reported symptoms together,"
    ]
}

pseudo_cough_sentences_prefixed = {
    2: ["From this cough sound, the characteristics are dry, persistent, moderate number of coughs, mild hoarseness, intermittent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, hacking, frequent coughs, no hoarseness, persistent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, moderate frequency, mild hoarseness, persistent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, occasional coughs, no hoarseness, intermittent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, persistent, frequent, slight hoarseness, persistent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, hacking, moderate number of coughs, mild hoarseness, persistent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, persistent, few coughs, no hoarseness, intermittent pattern; this indicates COVID-19.",
        "From this cough sound, the characteristics are dry, persistent, moderate frequency, mild hoarseness, persistent pattern; this indicates COVID-19."],
    1: ["From this cough sound, the characteristics are wet, persistent, moderate number of coughs, mild hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, hacking, frequent coughs, mild hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, persistent, many coughs, moderate hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, hacking, few coughs, mild hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, persistent, moderate frequency, moderate hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, hacking, frequent coughs, mild hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, persistent, many coughs, moderate hoarseness, persistent pattern; this indicates TB.",
        "From this cough sound, the characteristics are wet, hacking, moderate number of coughs, mild hoarseness, persistent pattern; this indicates TB."],
    0: ["From this cough sound, the characteristics are occasional, mild, no hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are rare, mild, no hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are occasional, non-productive, mild hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are intermittent, mild, no hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are rare, non-specific, no hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are mild, occasional, no hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are intermittent, non-productive, mild hoarseness, intermittent pattern; this cannot determine a specific disease.",
        "From this cough sound, the characteristics are occasional, mild, no hoarseness, intermittent pattern; this cannot determine a specific disease."]
}

DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets"
DESTINATION_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets"

system_prompt = '''You are an advanced medical assistant AI specialized in analyzing and diagnosing clinical conditions. 
You can interpret auditory inputs, visual inputs, and patient symptoms, individually or in combination. 
Some inputs may be missing, and you should reason based on whatever is provided. 

Your task is to:
1. Analyze the available input(s): X-ray images if present, cough audio if present, and reported symptoms if present.   
2. Explain your reasoning step by step.  
3. Give the most likely diagnosis.  

Always respond in the following structured format, using the special tokens exactly as shown:

<think>
Analyze the available data, explain your reasoning, and show how you arrived at your conclusion.
</think>

<answer>
State the most likely disease or diagnosis.
</answer>'''

for split in ['train', 'val']:
    df1 = pd.read_csv(f"{DATA_PATH}/coda/metadata_llm.csv.{split}").rename(columns={"participant": "user_id", "tb_status": "disease", "path_file": "path_file_audio"})
    df1 = df1[['user_id', 'disease', 'path_file_audio','symptoms_sentence', 'llm_analyze_symptoms']]
    df1 = df1.groupby("user_id", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 70), random_state=42))

    df2 = pd.read_csv(f"{DATA_PATH}/ukcovid/metadata_llm.csv.{split}").rename(columns={"participant_identifier": "user_id", "covid_test_result": "disease"})
    df2["disease"] = df2["disease"].replace(1, 2)
    df2 = df2[['user_id', 'disease', 'path_file_audio','symptoms_sentence', 'llm_analyze_symptoms']] 

    df3 = pd.read_csv(f"{DATA_PATH}/xray_metadata_llm_final.csv.{split}").rename(columns={"path_file": "path_file_image", "formatted_llm_analyze_image": "llm_analyze_image"})
    
    df_combined = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    if split == "val":
        split = "validation"
        df_balanced = sample_modalities_numeric_overlap(df_combined)
        df_balanced.reset_index(drop=True, inplace=True)
        df_balanced.to_csv(f"{DESTINATION_PATH}/{split}_balancediseaseaugment_raw.csv", index=False)
    elif split == "train":
        df_balanced = sample_modalities_numeric(df_combined, split, maximum_disease=10000)
        df_balanced.to_csv(f"{DESTINATION_PATH}/{split}_balancediseaseaugment_raw.csv", index=False)

    print(df_balanced.groupby(['disease', 'modalities_simple']).size())
    total_rows = len(df_balanced)
    num_duplicates = df_balanced.duplicated(subset=['path_file_audio', 'path_file_image'], keep='first').sum()
    dup_percentage = num_duplicates / total_rows * 100
    print(f"Total rows: {total_rows}")
    print(f"Duplicated rows: {num_duplicates}")
    print(f"Duplication percentage: {dup_percentage:.2f}%")

    instruct_array = []
    for now_row in tqdm(df_balanced.itertuples(), desc=f"Processing {split}", total=len(df_balanced)):
        if pd.isna(now_row.path_file_audio) == False and os.path.exists(
            "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets/" +
                now_row.path_file_audio
        ) == False:
            continue

        if pd.isna(now_row.path_file_image) == False and os.path.exists(
            "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets/" +
                now_row.path_file_image
        ) == False:
            continue

        key = now_row.modalities_simple
        if key in prompt_templates:
            last_sentence_question = random.choice(
                prompt_templates[key]) + " Determine what disease this could indicate, or answer 'Others' if you cannot determine."

        question = ""
        if not pd.isna(now_row.path_file_audio):
            question += "<audio>"
        if not pd.isna(now_row.path_file_image):
            question += "<image>"
        question += last_sentence_question # Harusnya kan ditaruh di akhir

        think_sentence = "Okay, let's see. "
        if not pd.isna(now_row.llm_analyze_symptoms):
            question += f"The patient symptoms are {now_row.symptoms_sentence}."
            think_sentence += now_row.llm_analyze_symptoms + " "
        if not pd.isna(now_row.path_file_audio):
            think_sentence += random.choice(
                pseudo_cough_sentences_prefixed[now_row.disease]) + " "
        if not pd.isna(now_row.llm_analyze_image):
            think_sentence += now_row.llm_analyze_image + " "
        think_sentence += f"Therefore, the disease is {disease_map_reverse[now_row.disease]}."
        answer = f"<think>{think_sentence}</think>\n\n<answer>{disease_map_reverse[now_row.disease]}</answer>"

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

        if not pd.isna(now_row.path_file_audio):
            filename_audio = now_row.path_file_audio.split(
                "/")[-1].split(".")[0]
            start_sec, end_sec = commons.random_3sec_segment(
                "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets/" + now_row.path_file_audio,
                segment_duration=3.0)
            temp_instruct["messages"][1]['content'].append({
                "type": "audio",
                "audio": f"{filename_audio}.wav",
                "audio_start": float(start_sec),
                "audio_end": float(end_sec)
            })
            temp_instruct["audio_file_name"] = f"{split}/{filename_audio}.wav"
        if not pd.isna(now_row.path_file_image):
            filename_image = now_row.path_file_image.split(
                "/")[-1].split(".")[0]
            temp_instruct["messages"][1]['content'].append(
                {"type": "image", "image": now_row.path_file_image.split("/")[-1]})
            temp_instruct["image_file_name"] = split + \
                "/" + now_row.path_file_image.split("/")[-1]

        temp_instruct["solution"] = answer
        temp_instruct["modalities"] = now_row.modalities_simple
        instruct_array.append(temp_instruct)

    print(len(instruct_array))
    df_result = pd.DataFrame(instruct_array)
    df_result.to_csv(f"{DESTINATION_PATH}/{split}_balancediseaseaugment_nodup.csv", index=False)

# for split in ['train', 'val']:
#     df1 = pd.read_csv(f"{DATA_PATH}/coda/metadata_llm.csv.{split}").rename(columns={"participant": "user_id", "tb_status": "disease", "path_file": "path_file_audio"})
#     df1 = df1[['user_id', 'disease', 'path_file_audio', 'symptoms_sentence', 'llm_analyze_symptoms']]
#     df1["disease"] = df1["disease"].replace(0, 4)
#     df1 = df1.groupby("user_id", group_keys=False).apply(
#         lambda x: x.sample(n=min(len(x), 70), random_state=42)
#     )
#     df2 = pd.read_csv(f"{DATA_PATH}/ukcovid/metadata_llm.csv.{split}").rename(columns={"participant_identifier": "user_id", "covid_test_result": "disease"})
#     df2["disease"] = df2["disease"].replace(1, 2)
#     df2["disease"] = df2["disease"].replace(0, 4)
#     df2 = df2[['user_id', 'disease', 'path_file_audio', 'symptoms_sentence', 'llm_analyze_symptoms']]
#     df3 = pd.read_csv(f"{DATA_PATH}/xray_metadata_llm_final.csv.{split}").rename(columns={"path_file": "path_file_image", "formatted_llm_analyze_image": "llm_analyze_image"})
#     df_combined = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

#     instruct_array = []
#     for now_row in tqdm(df_combined.itertuples(), desc=f"Processing {split}", total=len(df_combined)):
#         row_dict = now_row._asdict()
#         modalities = []
#         modalities_tags = ""
#         if not pd.isna(now_row.path_file_audio):
#             modalities_tags += "<audio>"
#             modalities.append("audio")
#         if not pd.isna(now_row.path_file_image):
#             modalities_tags += "<image>"
#             modalities.append("xray")
#         if not pd.isna(now_row.symptoms_sentence):
#             modalities.append("symptoms")

#         key = tuple(modalities)
#         if key in prompt_templates:
#             last_sentence_question = random.choice(prompt_templates[key]) + ","

#         question = ""
#         think_sentence = "Okay, let's see. "
#         if not pd.isna(now_row.llm_analyze_symptoms):
#             question += f"The patient symptoms are {now_row.symptoms_sentence}."
#             think_sentence += now_row.llm_analyze_symptoms + " "
#         if not pd.isna(now_row.llm_analyze_image):
#             think_sentence += now_row.llm_analyze_image + " "
#         think_sentence += f"Therefore, the disease is {disease_map_reverse[now_row.disease]}."
#         answer = f"<think>{think_sentence}</think>\n\n<answer>{disease_map_reverse[now_row.disease]}</answer>"

#         question = modalities_tags + "" + question
#         question += last_sentence_question

#         temp_instruct = {"messages": [
#             {"role": "system",
#                 "content": [
#                     {"type": "text", "text": system_prompt}
#                 ],
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question},
#                 ],
#             },
#             {"role": "assistant", "content": [
#                 {"type": "text", "text": answer}]},
#         ]}

#         if not pd.isna(now_row.path_file_audio):
#             os.makedirs(f"{DESTINATION_PATH}/{split}/", exist_ok=True)
#             filename_audio = now_row.path_file_audio.split("/")[-1].split(".")[0]
#             start_sec, end_sec = commons.random_3sec_segment(now_row.path_file_audio, segment_duration=3.0)
#             temp_instruct["messages"][1]['content'].append({
#                 "type": "audio",
#                 "audio": f"{filename_audio}.wav",
#                 "audio_start": float(start_sec),
#                 "audio_end": float(end_sec)
#             })
#             #temp_instruct["audios"] = now_row.path_file_audio
#             shutil.copy(now_row.path_file_audio, f"{DESTINATION_PATH}/{split}/{filename_audio}.wav")
#             temp_instruct["audio_file_name"] = f"{filename_audio}.wav"
#         if not pd.isna(now_row.path_file_image):
#             filename_image = now_row.path_file_image.split("/")[-1].split(".")[0]
#             temp_instruct["messages"][1]['content'].append({"type": "image", "image": now_row.path_file_image.split("/")[-1]})
#             #temp_instruct["images"] = now_row.path_file_image
#             os.makedirs(f"{DESTINATION_PATH}/{split}/", exist_ok=True)
#             shutil.copy(now_row.path_file_image, f"{DESTINATION_PATH}/{split}/{now_row.path_file_image.split('/')[-1]}")
#             temp_instruct["image_file_name"] = now_row.path_file_image.split("/")[-1]

#         temp_instruct["solution"] = answer
#         temp_instruct["identifier"] = now_row.user_id
#         instruct_array.append(temp_instruct)

#     df_result = pd.DataFrame(instruct_array)
#     df_result.to_csv(f"{DESTINATION_PATH}/{split}.csv", index=False)

# from huggingface_hub import HfApi
# api = HfApi()

# print("Uploading Datasets.......")
# api.upload_large_folder(
#     repo_id="arkiven4/grpo_3modalities_datasets",
#     repo_type="dataset",
#     folder_path="/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets",
# )

# hf upload-large-folder arkiven4/grpo_3modalities_datasets --repo-type=dataset /home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets --num-workers=16 --private

# def sample_modalities_numericv1(df, seed=42):
#     random.seed(seed)
#     np.random.seed(seed)

#     for col in ['path_file_audio', 'path_file_image']:
#         df[col] = df[col].apply(lambda x: os.path.join(
#             'validation', os.path.basename(x)) if pd.notna(x) else np.nan)

#     augmented_rows = []

#     df = df[df['disease'].isin([1, 2, 4])].reset_index(drop=True)
#     disease_classes = df['disease'].unique()

#     for label in tqdm(disease_classes, desc="Processing disease classes"):
#         class_df = df[df['disease'] == label].reset_index(drop=True)

#         audio_rows = class_df[~class_df['path_file_audio'].isna()]
#         image_rows = class_df[~class_df['path_file_image'].isna()]
#         symptoms_rows = class_df[~class_df['symptoms_sentence'].isna()]

#         combo_to_rows = {combo: [] for combo in [
#             ("path_file_audio",),
#             ("path_file_image",),
#             ("symptoms_sentence",),
#             ("path_file_audio", "path_file_image"),
#             ("path_file_audio", "symptoms_sentence"),
#             ("path_file_image", "symptoms_sentence"),
#             ("path_file_audio", "path_file_image", "symptoms_sentence"),
#         ]}

#         for _, row in class_df.iterrows():
#             base_modalities = {
#                 "path_file_audio": row['path_file_audio'] if pd.notna(row['path_file_audio']) else None,
#                 "path_file_image": row['path_file_image'] if pd.notna(row['path_file_image']) else None,
#                 "symptoms_sentence": row['symptoms_sentence'] if pd.notna(row['symptoms_sentence']) else None
#             }
#             all_combos = [
#                 ("path_file_audio",),
#                 ("path_file_image",),
#                 ("symptoms_sentence",),
#                 ("path_file_audio", "path_file_image"),
#                 ("path_file_audio", "symptoms_sentence"),
#                 ("path_file_image", "symptoms_sentence"),
#                 ("path_file_audio", "path_file_image", "symptoms_sentence"),
#             ]
#             for combo in all_combos:
#                 new_row = row.to_dict()
#                 for m in combo:
#                     if base_modalities[m] is None:
#                         if m == "path_file_audio" and len(audio_rows) > 0:
#                             new_row[m] = audio_rows.sample(
#                                 1, random_state=seed).iloc[0]['path_file_audio']
#                         elif m == "path_file_image" and len(image_rows) > 0:
#                             new_row[m] = image_rows.sample(
#                                 1, random_state=seed).iloc[0]['path_file_image']
#                         elif m == "symptoms_sentence" and len(symptoms_rows) > 0:
#                             new_row[m] = symptoms_rows.sample(
#                                 1, random_state=seed).iloc[0]['symptoms_sentence']
#                     else:
#                         new_row[m] = base_modalities[m]
#                 new_row["modalities_used"] = combo
#                 combo_to_rows[combo].append(new_row)

#         min_count = min(len(rows)
#                         for rows in combo_to_rows.values() if len(rows) > 0)
#         for combo, rows in combo_to_rows.items():
#             if not rows:
#                 continue
#             sampled_rows = random.sample(rows, min_count)
#             augmented_rows.extend(sampled_rows)

#     final_df = pd.DataFrame(augmented_rows).sample(
#         frac=1, random_state=seed).reset_index(drop=True)
#     return final_df
