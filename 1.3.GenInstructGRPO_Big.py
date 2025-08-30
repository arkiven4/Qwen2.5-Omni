import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
import os, librosa, random, pickle, pydicom, requests, torch, re, shutil
from pydicom.datadict import keyword_for_tag
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from IPython.display import Markdown, display
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

import commons, const_variable

from huggingface_hub import login
from dotenv import load_dotenv
from datasets import Dataset, concatenate_datasets

load_dotenv()
login(token=os.getenv("HG_AUTH_KEY_WRITE"))

random.seed(42)

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
        "Does the cough audio indicate",
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
        "Do the presented symptoms indicate",
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
        "Do the symptoms and cough sound indicate",
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


DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets"
DESTINATION_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets"

system_prompt = '''You are an advanced medical assistant AI specialized in analyzing and diagnosing clinical conditions, capable of perceiving auditory and visual inputs.
You can interpret and reason over various medical inputs, including auditory inputs, visual inputs, and patient symptoms, individually or in combination, depending on what is provided.
Your task is to analyze the given input, explain your reasoning, and give a possible diagnosis.
Always respond in the following format:
<think>{Analyze the available data: X-ray findings if present, cough audio characteristics if provided, and reported symptoms to determine the most likely disease.}</think>
<answer>{Determine the disease}</answer>'''

for split in ['train', 'val']:
    df1 = pd.read_csv(f"{DATA_PATH}/coda/metadata_llm.csv.{split}").rename(columns={"participant": "user_id", "tb_status": "disease", "path_file": "path_file_audio"})
    df1 = df1[['user_id', 'disease', 'path_file_audio', 'symptoms_sentence', 'llm_analyze_symptoms']]
    df1["disease"] = df1["disease"].replace(0, 4)
    df1 = df1.groupby("user_id", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 70), random_state=42)
    )
    df2 = pd.read_csv(f"{DATA_PATH}/ukcovid/metadata_llm.csv.{split}").rename(columns={"participant_identifier": "user_id", "covid_test_result": "disease"})
    df2["disease"] = df2["disease"].replace(1, 2)
    df2["disease"] = df2["disease"].replace(0, 4)
    df2 = df2[['user_id', 'disease', 'path_file_audio', 'symptoms_sentence', 'llm_analyze_symptoms']]
    df3 = pd.read_csv(f"{DATA_PATH}/xray_metadata_llm_final.csv.{split}").rename(columns={"path_file": "path_file_image", "formatted_llm_analyze_image": "llm_analyze_image"})
    df_combined = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    instruct_array = []
    for now_row in tqdm(df_combined.itertuples(), desc=f"Processing {split}", total=len(df_combined)):
        row_dict = now_row._asdict()
        modalities = []
        modalities_tags = ""
        if not pd.isna(now_row.path_file_audio):
            modalities_tags += "<audio>"
            modalities.append("audio")
        if not pd.isna(now_row.path_file_image):
            modalities_tags += "<image>"
            modalities.append("xray")
        if not pd.isna(now_row.symptoms_sentence):
            modalities.append("symptoms")

        key = tuple(modalities)
        if key in prompt_templates:
            last_sentence_question = random.choice(prompt_templates[key]) + ", Determine what disease this could indicate, or answer 'Others' if you cannot determine."
        
        question = ""
        think_sentence = "Okay, let's see. "
        if not pd.isna(now_row.llm_analyze_symptoms):
            question += f"The patient symptoms are {now_row.symptoms_sentence}."
            think_sentence += now_row.llm_analyze_symptoms + " "
        if not pd.isna(now_row.llm_analyze_image):
            think_sentence += now_row.llm_analyze_image + " "
        think_sentence += f"Therefore, the disease is {disease_map_reverse[now_row.disease]}."
        answer = f"<think>{think_sentence}</think>\n\n<answer>{disease_map_reverse[now_row.disease]}</answer>"

        question = modalities_tags + "" + question
        question += last_sentence_question

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
            os.makedirs(f"{DESTINATION_PATH}/{split}/", exist_ok=True)
            filename_audio = now_row.path_file_audio.split("/")[-1].split(".")[0]
            start_sec, end_sec = commons.random_3sec_segment(now_row.path_file_audio, segment_duration=3.0)
            temp_instruct["messages"][1]['content'].append({
                "type": "audio", 
                "audio": f"{filename_audio}.wav",
                "audio_start": float(start_sec),
                "audio_end": float(end_sec)
            })
            #temp_instruct["audios"] = now_row.path_file_audio
            shutil.copy(now_row.path_file_audio, f"{DESTINATION_PATH}/{split}/{filename_audio}.wav")
            temp_instruct["audio_file_name"] = f"{filename_audio}.wav"
        if not pd.isna(now_row.path_file_image):
            filename_image = now_row.path_file_image.split("/")[-1].split(".")[0]
            temp_instruct["messages"][1]['content'].append({"type": "image", "image": now_row.path_file_image.split("/")[-1]})
            #temp_instruct["images"] = now_row.path_file_image
            os.makedirs(f"{DESTINATION_PATH}/{split}/", exist_ok=True)
            shutil.copy(now_row.path_file_image, f"{DESTINATION_PATH}/{split}/{now_row.path_file_image.split('/')[-1]}")
            temp_instruct["image_file_name"] = now_row.path_file_image.split("/")[-1]

        temp_instruct["solution"] = answer
        temp_instruct["identifier"] = now_row.user_id
        instruct_array.append(temp_instruct)

    df_result = pd.DataFrame(instruct_array)
    df_result.to_csv(f"{DESTINATION_PATH}/{split}.csv", index=False)

from huggingface_hub import HfApi
api = HfApi()

print("Uploading Datasets.......")
api.upload_large_folder(
    repo_id="arkiven4/grpo_3modalities_datasets",
    repo_type="dataset",
    folder_path="/home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets",
)

#hf upload-large-folder arkiven4/grpo_3modalities_datasets --repo-type=dataset /home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets --num-workers=16 --private