import json
import re
import requests
import pydicom
import pickle
import random
import librosa
import const_variable
import commons
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

device_name = torch.cuda.get_device_name(0)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"Logical index: {i}, Name: {props.name}")


DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets/ukcovid"
TARGET_COL = "covid_test_result"
IDX_COL = "participant_identifier"
model_id = "/home/is/dwipraseetyo-a/NAS_HAI/Project/pretrain/Llama3-OpenBioLLM-8B"
system_prompt = "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="bfloat16",
    device_map="auto"
)

model.generation_config.pad_token_id = tokenizer.pad_token_id
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
model.eval()

for split in ['train', 'val', 'test']:
    df = pd.read_csv(f"{DATA_PATH}/metadata_llm.csv.{split}")
    df_unique = df.groupby(IDX_COL).first().reset_index()

    results = []
    participant_ident = []
    messages_batch = []
    for now_row in tqdm(df_unique.itertuples(), desc=f"Processing {split}", total=len(df_unique)):
        row_dict = now_row._asdict()
        answer = "Positive COVID-19" if row_dict[TARGET_COL] == 1 else "Negative COVID-19"
        symptoms_sentence = row_dict['symptoms_sentence']
        messages_batch.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The Patient Symptoms Are: {symptoms_sentence}. Explain to me why this patient is {answer}?."},
        ])
        participant_ident.append(row_dict[IDX_COL])

    batch_size = 256
    for i in tqdm(range(0, len(messages_batch), batch_size), desc="Generating", total=len(range(0, len(messages_batch), batch_size))):
        batch_messages = messages_batch[i:i+batch_size]
        batch_idxs = participant_ident[i:i+batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            for msgs in batch_messages
        ]

        prompt_ids_list = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]
        prompt_lengths = [len(ids) for ids in prompt_ids_list]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )

        for j, (outputs_id) in enumerate(outputs_ids):
            response_ids = outputs_id[prompt_lengths[j]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
            results.append({
                "idx_col": batch_idxs[j],
                "llm_analyze_symptoms": response_text
            })
        del outputs_ids
        torch.cuda.empty_cache()
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{DATA_PATH}/asdasd.csv.{split}", index=False)
    merged = df.merge(
        results_df.rename(columns={"idx_col": IDX_COL}),
        on=IDX_COL,
        how="left"
    )
    merged.to_csv(f"{DATA_PATH}/metadata_llm.csv.{split}", index=False)