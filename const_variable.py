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
        "Based on the provided cough audio, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Listen to this cough sound, is it Positive Tuberculosis or Negative Tuberculosis?",
        "Does the cough audio indicate Positive Tuberculosis or Negative Tuberculosis?",
        "Analyze the following cough sound. Is this case Positive Tuberculosis or Negative Tuberculosis?"
    ],
    ("xray",): [
        "Examine this chest x-ray, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Does the provided x-ray image show Positive Tuberculosis or Negative Tuberculosis?",
        "Analyze the radiographic scan, is it Positive Tuberculosis or Negative Tuberculosis?",
        "From this x-ray image, is this case Positive Tuberculosis or Negative Tuberculosis?"
    ],
    ("symptoms",): [
        "Given these symptoms, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Do the presented symptoms indicate Positive Tuberculosis or Negative Tuberculosis?",
        "Analyze the following symptoms, is it Positive Tuberculosis or Negative Tuberculosis?",
        "Based on the symptom description, is this case Positive Tuberculosis or Negative Tuberculosis?"
    ],
    ("audio", "xray"): [
        "Based on the chest x-ray and cough audio, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Examine the x-ray and cough sound, is it Positive Tuberculosis or Negative Tuberculosis?",
        "Given the radiograph and cough recording, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Using both the x-ray and cough audio, is this Positive Tuberculosis or Negative Tuberculosis?"
    ],
    ("audio", "symptoms"): [
        "Given the cough audio and symptoms, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Analyze the patient symptoms and cough, is it Positive Tuberculosis or Negative Tuberculosis?",
        "Do the symptoms and cough sound indicate Positive Tuberculosis or Negative Tuberculosis?",
        "Using both the cough sound and symptoms, is this case Positive Tuberculosis or Negative Tuberculosis?"
    ],
    ("xray", "symptoms"): [
        "From the x-ray and symptoms, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Analyze the x-ray and clinical symptoms, is it Positive Tuberculosis or Negative Tuberculosis?",
        "Given the chest scan and symptoms, is this Positive Tuberculosis or Negative Tuberculosis?",
        "What is the diagnosis based on the x-ray and symptoms, Positive Tuberculosis or Negative Tuberculosis?"
    ],
    ("audio", "xray", "symptoms"): [
        "Based on the x-ray, cough audio, and symptoms, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Evaluate the x-ray, cough sound, and symptoms, is it Positive Tuberculosis or Negative Tuberculosis?",
        "Given the combined evidence, is this case Positive Tuberculosis or Negative Tuberculosis?",
        "Considering all inputs (image, sound, and symptoms), is this Positive Tuberculosis or Negative Tuberculosis?"
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

system_prompt = (
    "A conversation between User and Advanced medical assistant specialized in analyzing and diagnosing clinical conditions. and the Assistant determines whether the case is Positive or Negative. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)