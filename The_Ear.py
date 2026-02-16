import whisper
import streamlit as st


@st.cache_resource
def Loadmodel():
    model = whisper.load_model("small.en")
    return model


def get_audio_data(file_path):
    model = Loadmodel()
    result = model.transcribe(file_path)
    audio_data = []
    for segment in result["segments"]:
        item = {
            "start": segment["start"],
            "text": segment["text"],
        }
        audio_data.append(item)
    return audio_data
