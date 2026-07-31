import streamlit as st
import os
import torch
import subprocess
import numpy as np
from transformers import pipeline
import re

@st.cache_resource
def Loadmodel():
    model_dir = "Minh1506/cortex-whisper"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        tokenizer=model_dir,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
    )
    return pipe

def load_audio(file_path: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary.
    This replaces openai-whisper's load_audio and avoids transformers' stdin bug!
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # Run ffmpeg and capture the output
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    
    # Convert from 16-bit PCM bytes to float32 numpy array
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def get_audio_data(file_path, unique_key):
    pipe = Loadmodel()
    
    # Load the audio into a numpy array using ffmpeg directly.
    # This prevents the HuggingFace pipeline from failing on .m4a and .mp4 files
    audio_array = load_audio(file_path)
    
    # Pass the raw array to the pipeline
    result = pipe({"array": audio_array, "sampling_rate": 16000})
    
    audio_data = []
    
    for chunk in result["chunks"]:
        start_time, end_time = chunk["timestamp"]
        
        if end_time is None:
            end_time = start_time + 5.0
            
        text = chunk["text"].strip()
        if not text:
            continue
            
        # Split the text at periods, question marks, or exclamation points.
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        # If there is literally NO punctuation, we can forcefully chop it every 100 characters
        if len(sentences) == 1 and len(text) > 100:
            sentences = [text[i:i+100] for i in range(0, len(text), 100)]
            
        total_length = len(text)
        total_duration = end_time - start_time
        
        current_start = start_time
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: 
                continue
                
            # Estimate how many seconds this sentence took based on character length
            proportion = len(sentence) / max(total_length, 1)
            duration = total_duration * proportion
            current_end = current_start + duration
            
            item = {
                "start": current_start,
                "end": current_end,
                "text": "Text: " + sentence,
                "file_id": unique_key,
            }
            audio_data.append(item)
            
            # Move the start time forward for the next sentence
            current_start = current_end

    # Create stuff for Acollection
    Alist_ids = []
    Alist_document = []
    Alist_metadatas = []
    n = 1
    for i in audio_data:
        Alist_ids.append(f"{unique_key}_{n}")
        Alist_document.append(i["text"])
        item = {"start": i["start"], "end": i["end"], "file_id": i["file_id"]}
        Alist_metadatas.append(item)
        n += 1
        
    return Alist_ids, Alist_document, Alist_metadatas
