import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Set environment variables for distributed training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"

import commons, const_variable
import librosa, random, pickle, pydicom, requests, re, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def setup(rank, world_size):
    """Initialize the process group"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def crop_and_convert(path_file, max_size=768):
    array_PIL = Image.open(path_file)
    
    # Resize if needed
    width, height = array_PIL.size
    max_dim = max(width, height)
    
    if max_dim > max_size:
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        array_PIL = array_PIL.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    
    return array_PIL

def get_dataloader_for_rank(df, rank, world_size):
    """Split dataframe across ranks"""
    total_samples = len(df)
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    
    if rank == world_size - 1:  # Last rank gets remaining samples
        end_idx = total_samples
    else:
        end_idx = start_idx + samples_per_rank
    
    return df.iloc[start_idx:end_idx]

def process_split(rank, world_size, split):
    """Process a single split on a specific GPU"""
    setup(rank, world_size)
    
    disease_map = {
        "Normal": 0,
        "Tuberculosis": 1,
        "Covid-19": 2,
        "Pneumonia": 3,
        "Other": 4,
    }
    reverse_map = {v: k for k, v in disease_map.items()}

    DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets"
    TARGET_COL = "disease"
    IDX_COL = "user_id"

    model_id = "/home/is/dwipraseetyo-a/NAS_HAI/Project/pretrain/medgemma-4b-it"
    auth_key = os.getenv("HG_AUTH_KEY")
    
    # Load model on current GPU
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
        token=auth_key,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    model.eval()

    # Load and split data
    df = pd.read_csv(f"{DATA_PATH}/xray_metadata_llm.csv.{split}")
    rank_df = get_dataloader_for_rank(df, rank, world_size)
    
    if rank == 0:
        print(f"Processing {split} split with {len(df)} total samples")
        print(f"Rank {rank} processing {len(rank_df)} samples")

    results = []
    participant_idents = []
    messages_batch = []
    
    # Process samples assigned to this rank
    for now_row in tqdm(rank_df.itertuples(), 
                       desc=f"Rank {rank} - Processing {split}", 
                       total=len(rank_df),
                       disable=(rank != 0)):  # Only show progress on rank 0
        row_dict = now_row._asdict()
        now_image = crop_and_convert(now_row.path_file)
        answer = f"This is {reverse_map[row_dict[TARGET_COL]]} Disease"
        
        system_prompt = (
            "You are an expert radiologist. Analyze the chest X-ray and return only the following sections:\n\n"
            "**Specific Findings:** Provide a comprehensive and detailed description of all relevant radiographic abnormalities.\n\n"
            "**Differential Diagnosis:** Only if the X-ray not healty and describe the reason.\n\n"
            "**Conclusion:** Summarize your diagnostic impression clearly.\n\n"
            "Do not include any **Disclaimer** or unrelated content."
        ) if row_dict[TARGET_COL] == 0 else (
            "You are an expert radiologist. Analyze the chest X-ray and return only the following sections:\n\n"
            "**Key Features:** Highlight radiographic patterns that are characteristic of disease, explained clearly and in detail.\n\n"
            "**Conclusion:** Summarize your diagnostic impression clearly.\n\n"
            "Do not include any **Disclaimer** or unrelated content."
        )
        
        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"Describe the X-ray." + answer},
                {"type": "image", "image": now_image}
            ]}
        ]
        messages_batch.append(message)
        participant_idents.append(row_dict[IDX_COL])
    

    # Process batches
    batch_size = 8  # Reduced batch size for distributed processing
    device = torch.device(f"cuda:{rank}")
    
    for i in tqdm(range(0, len(messages_batch), batch_size), 
                  desc=f"Rank {rank} - Generating", 
                  total=len(range(0, len(messages_batch), batch_size)),
                  disable=(rank != 0)):
        batch_messages = messages_batch[i:i+batch_size]
        batch_idents = participant_idents[i:i+batch_size]

        inputs = processor.apply_chat_template(
            batch_messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt", 
            padding=True
        ).to(device, dtype=torch.bfloat16)

        input_lens = [len(ids) for ids in inputs['input_ids']]

        with torch.inference_mode():
            generation = model.module.generate(**inputs, max_new_tokens=768, do_sample=False)

        # Slice outputs to exclude input prompt
        for j in range(len(batch_messages)):
            output_ids = generation[j][input_lens[j]:]
            decoded = processor.decode(output_ids, skip_special_tokens=True)
            results.append({
                "idx_col": batch_idents[j],
                "llm_analyze_image": decoded
            })

        del generation
        torch.cuda.empty_cache()

    # Save results for this rank
    results_df = pd.DataFrame(results)
    rank_output_file = f"{DATA_PATH}/xray_medgemma_rank{rank}.csv.{split}"
    results_df.to_csv(rank_output_file, index=False)
    
    if rank == 0:
        print(f"Rank {rank} saved {len(results)} results to {rank_output_file}")
    
    cleanup()
    return rank_output_file

def combine_results(world_size, split):
    """Combine results from all ranks"""
    DATA_PATH = "/home/is/dwipraseetyo-a/NAS_HAI/Datasets"
    IDX_COL = "user_id"
    
    # Load original dataframe
    df = pd.read_csv(f"{DATA_PATH}/xray_metadata_llm.csv.{split}")
    
    # Combine results from all ranks
    all_results = []
    for rank in range(world_size):
        rank_file = f"{DATA_PATH}/xray_medgemma_rank{rank}.csv.{split}"
        if os.path.exists(rank_file):
            rank_results = pd.read_csv(rank_file)
            all_results.append(rank_results)
            # Clean up temporary files
            os.remove(rank_file)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{DATA_PATH}/xray_medgemma.csv.{split}", index=False)
        
        # Merge with original dataframe
        merged = df.merge(
            combined_results.rename(columns={"idx_col": IDX_COL}),
            on=IDX_COL,
            how="left"
        )
        merged.to_csv(f"{DATA_PATH}/xray_metadata_llm.csv.{split}", index=False)
        print(f"Combined results saved for {split} split: {len(combined_results)} samples")

def main():
    """Main function to run distributed processing"""
    world_size = 2  # Number of GPUs
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    if torch.cuda.device_count() < world_size:
        print(f"Only {torch.cuda.device_count()} GPUs available, need {world_size}")
        return
    
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Logical index: {i}, Name: {props.name}")
    
    # Process each split
    for split in ['train', 'test', 'val']:
        print(f"\n=== Processing {split} split ===")
        
        # Spawn processes for distributed processing
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=process_split, args=(rank, world_size, split))
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine results from all ranks
        combine_results(world_size, split)
        print(f"Completed processing {split} split")

if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn')
    main()