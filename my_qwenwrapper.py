import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig, StoppingCriteriaList
from commons import StopOnMultiToken

import warnings, logging
class SuppressQwenWarning(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("System prompt modified, audio output may not work as expected")

warnings.filterwarnings("ignore", message="System prompt modified, audio output may not work as expected. Audio output mode only works when using default system prompt 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'")
logging.getLogger().addFilter(SuppressQwenWarning())

def get_OmniModel(model_path="Qwen/Qwen2.5-Omni-3B", adapter_path=None, processor_path="Qwen/Qwen2.5-Omni-3B", 
                  padding_side=None, use_flash_attention=True, only_processor=False, quantize_4bit=True, 
                  offload_folder=None, set_eval=True):
    
    if padding_side == "left":
        print("Loading Processsor.... Using Left padding Side")
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path, padding_side="left")
    else:
        processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)

    if only_processor:
        return processor

    # if set_eval:
    #     quantize_4bit = False

    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
    if quantize_4bit:
        print("Loading Model.... Using BitsAndBytesConfig")
        bnb_config = BitsAndBytesConfig( load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        bnb_config = None

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if quantize_4bit else "auto",
        "device_map": "auto",
        "quantization_config": bnb_config,
        "attn_implementation": attn_implementation,
    }
    if offload_folder is not None:
        print("Loading Model.... Using Offload Folder")
        model_kwargs["offload_folder"] = offload_folder
    if use_flash_attention:
        print("Loading Model.... Using Flash Attention")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )

    if set_eval:
        # Gausah Kuantisasi
        model.load_adapter(adapter_path) # outputs/qwen25omni-3b-instructMedic-Reason-notallpresent-llmimagellmsymp-trl-sft/checkpoint-3000
        model.eval()

    return model, processor

def get_stoppingcrit(processor):
    stop_token = "<|im_end|>"
    stop_token_ids = processor(stop_token, add_special_tokens=False)["input_ids"][0]
    stopping_criteria = StoppingCriteriaList([StopOnMultiToken(stop_token_ids)])

    return stopping_criteria