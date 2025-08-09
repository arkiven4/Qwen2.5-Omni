import torch
from torch.utils.data import Dataset, DataLoader

import commons
import const_variable

class QwenOmniFinetuneDataset(Dataset):
    def __init__(self, data, processor, use_audio_in_video=False):
        self.data = data
        self.processor = processor
        self.use_audio_in_video = use_audio_in_video

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data[idx]["messages"]
        for ele in conversation[1]['content']:
            if ele["type"] == "audio":
                if "audio" in ele or "audio_url" in ele:
                    path = ele.get("audio", ele.get("audio_url"))
                    start_sec, end_sec = commons.random_3sec_segment(path, segment_duration=3.0)
                    ele["audio_start"] = float(start_sec)
                    ele["audio_end"] = float(end_sec)

        return conversation