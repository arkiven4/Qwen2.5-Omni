from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

clientOpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

columns_imagefeat = ['Modality',
       'DistanceSourceToDetector', 'FieldOfViewDimensions',
       'ImagerPixelSpacing', 'TargetExposureIndex', 'Sensitivity',
       'PhotometricInterpretation', 'Rows', 'Columns',
       'PixelIntensityRelationshipSign', 'WindowCenter', 'WindowWidth',
       'RescaleIntercept', 'RescaleSlope']

columns_soundfeat = ['sex', 'cough_duration', 'cough_productive', 'haemoptysis', 'chestpain', 'shortbreath', 'fever',
'night_sweets', 'weight_loss', 'hiv_status', 'tobacco_use', 'cigarretes_perday', 'body_weight', 'height', 'bmi']

prompt_templates = {
    ("audio",): [
        "Based on the provided cough audio, could this be tuberculosis?",
        "Listen to this cough sound, is it consistent with tuberculosis?",
        "Does the cough audio suggest the presence of tuberculosis?",
        "Analyze the following cough sound. Is this indicative of TB?"
    ],
    ("xray",): [
        "Examine this chest x-ray. Are there signs suggestive of tuberculosis?",
        "Does the provided x-ray image show evidence of tuberculosis?",
        "Analyze the radiographic scan, could this be tuberculosis?",
        "From this x-ray image, is there any indication of pulmonary TB?"
    ],
    ("symptoms",): [
        "Given these symptoms, is tuberculosis a likely diagnosis?",
        "Do the presented symptoms match with tuberculosis?",
        "Analyze the following symptoms, could this be TB?",
        "Based on the symptom description, is this tuberculosis?"
    ],
    ("audio", "xray"): [
        "Based on the chest x-ray and cough audio, is this tuberculosis?",
        "Examine the x-ray and cough sound, could this indicate TB?",
        "Given the radiograph and cough recording, is tuberculosis likely?",
        "Using both the x-ray and cough audio, does this point to TB?"
    ],
    ("audio", "symptoms"): [
        "Given the cough audio and symptoms, is tuberculosis a possible diagnosis?",
        "Analyze the patient symptoms and cough, could this be TB?",
        "Do the symptoms and cough sound suggest tuberculosis?",
        "Using both the cough sound and symptoms, is this tuberculosis?"
    ],
    ("xray", "symptoms"): [
        "From the x-ray and symptoms, is this tuberculosis?",
        "Analyze the x-ray and clinical symptoms, do they indicate TB?",
        "Given the chest scan and symptoms, could this be pulmonary TB?",
        "What the likelihood this is tuberculosis based on the x-ray and symptoms?"
    ],
    ("audio", "xray", "symptoms"): [
        "Based on the x-ray, cough audio, and symptoms, is this tuberculosis?",
        "Evaluate the x-ray, cough sound, and symptoms, could this be TB?",
        "Given the combined evidence, does this case represent tuberculosis?",
        "Considering all inputs image, sound, and symptoms, is this TB?"
    ],
}

positive_templates = [
    "the diagnosis appears to be Tuberculosis.",
    "this case is likely indicative of Tuberculosis.",
    "signs point toward a Tuberculosis diagnosis.",
    "the evidence suggests the patient has Tuberculosis.",
    "Tuberculosis is a likely diagnosis here.",
    "this seems to be a case of Tuberculosis.",
    "the most probable condition is Tuberculosis.",
    "based on the findings, Tuberculosis is the most likely diagnosis."
]

negative_templates = [
    "the diagnosis does not seem to be Tuberculosis.",
    "it is unlikely that this is a case of Tuberculosis.",
    "this case appears to be non-Tuberculosis.",
    "no strong indications of Tuberculosis are found.",
    "the findings do not support a Tuberculosis diagnosis.",
    "this is likely not Tuberculosis.",
    "Tuberculosis is not the likely cause in this case.",
    "the patient probably does not have Tuberculosis."
]