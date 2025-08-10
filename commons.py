from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import const_variable
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import soundfile as sf
import copy
from qwen_omni_utils import process_mm_info

random.seed(42)

def pretty_status(message):
    line = "=" * (len(message) + 4)
    print(f"\n{line}")
    print(f"| {message} |")
    print(f"{line}\n")

def crop_and_convertNP(path_file):
    array = np.load(path_file)
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
    return array_PIL

def load_image_PIL(loaded_object):
    for obj in tqdm(loaded_object):
        for message in obj.get("messages", []):
            for content in message.get("content", []):
                if isinstance(content, dict) and "image" in content:
                    content["image"] = crop_and_convertNP(content["image"])
    return loaded_object

def crop_and_convert(path_file):
    array = np.load(path_file)
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
    return array_PIL

def random_3sec_segment(audio_path, segment_duration=3.0):
    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate
    if total_duration <= 3:
        return 0, segment_duration

    max_start = total_duration - segment_duration
    start_sec = np.round(np.random.uniform(0, max_start), 1)
    end_sec = start_sec + segment_duration
    return start_sec, end_sec

def unique_modalities_generator(prompt_templates):
    while True:
        modalities = []
        if random.random() < 0.5:
            modalities.append("audio")  
        if random.random() < 0.5:
            modalities.append("xray")
        if random.random() < 0.5:
            modalities.append("symptoms")

        if len(modalities) > 0:
            break

    key = tuple(modalities)
    if key in prompt_templates:
        prompt = random.choice(prompt_templates[key])

    try:
        return prompt, modalities
    except:
        return None, None

def grpo_build_datasets(instruct, processor):
    datasets = []
    for sample in tqdm(instruct):
        image_found = False
        audio_found = False
        conversation  = copy.deepcopy(sample["messages"])
        for ele in conversation[1]['content']:
            if ele["type"] == "audio":
                if "audio" in ele or "audio_url" in ele:
                    path = ele.get("audio", ele.get("audio_url"))
                    start_sec, end_sec = random_3sec_segment(path, segment_duration=3.0)
                    ele["audio_start"] = float(start_sec)
                    ele["audio_end"] = float(end_sec)
                    audio_found = True
            if ele["type"] == "image":
                if "image" in ele or "audio_url" in ele:
                    image_found = True

        solution = conversation[2]['content'][0]['text']
        del conversation[2]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        #audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        current_object = {
            "prompt": prompt,
            "solution": solution,
        }
        if images != None:
            current_object['image'] = images[0]   
        if audios != None:
            current_object['audio'] = audios[0]

        if image_found == False:
            continue
        if audio_found == False:
            continue 
        datasets.append({'messages' : current_object})

    return datasets

def get_prefix(modalities):
    parts = []
    if "symptoms" in modalities:
        parts.append("symptoms")
    if "audio" in modalities:
        parts.append("cough sound")
    if "xray" in modalities:
        parts.append("x-ray")

    if not parts:
        return "Based on the available data"
    
    if len(parts) == 1:
        return f"From the given {parts[0]}"
    elif len(parts) == 2:
        return f"From the given {parts[0]} and {parts[1]}"
    else:
        return f"From the given {parts[0]}, {parts[1]}, and {parts[2]}"

def generate_tb_response(modalities, llm_analyze_symptoms, llm_analyze_image, positive=True):
    llm_analyze_image = llm_analyze_image[2:]
    prefix = get_prefix(modalities) + ", Let me Analyze your regrading your questions.\n\n"
    #templates = const_variable.positive_templates if positive else const_variable.negative_templates
    sentence_tb = "Positive Tuberculosis" if positive else "Negative Tuberculosis" #random.choice(templates)

    missing_notes = []
    if "audio" not in modalities:
        missing_notes.append("* No Audio Present")
    if "xray" not in modalities:
        missing_notes.append("* No X-ray Image Provided")
    if "symptoms" not in modalities:
        missing_notes.append("* No Symptom Data Provided")

    review_message = "## ⚠️ Points to Review and Disclaimer\n"
    if missing_notes:
        review_message = review_message + "\n".join(missing_notes) + "\n\n"
    else:
        review_message = "*   All modalities are present.\n"  # or "All modalities are present." if you prefer
    review_message += "This is a preliminary interpretation based on given data and does not replace a comprehensive clinical evaluation. A definitive diagnosis requires a additional clinical evaluation, including the physical examination findings, Cough Sound, Auscultation Sound, and imaging studies.\n"

    #overview_message = f"## 🧠 Overview\n{sentence_tb}\n\n"
    orbservation_message = f"## 📋 Observations\n"
    if "symptoms" in modalities:
        orbservation_message += f"**Symptoms:**\n*   {llm_analyze_symptoms}\n\n"
    if "xray" in modalities:
        orbservation_message += f"**Chest X-ray:**\n{llm_analyze_image}\n\n"
    if "audio" in modalities:
        orbservation_message += f"**Audio:**\n*   Will be Implemented Soon"
    orbservation_message = orbservation_message.rstrip("\n")
    return f"<think>{prefix}{review_message}{orbservation_message}</think>\n\n<answer>{sentence_tb}</answer>"

class StopOnMultiToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids  # list of ints
        self.sequence_length = len(stop_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if input_ids.shape[1] < self.sequence_length:
            return False  # too early to match

        # Check if last N tokens match the stop sequence
        return input_ids[0, -self.sequence_length:].tolist() == self.stop_token_ids