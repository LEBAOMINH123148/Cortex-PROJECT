import whisper
import streamlit as st


@st.cache_resource
def Loadmodel():
    model = whisper.load_model("tiny")
    return model


def get_audio_data(file_path, unique_key):
    model = Loadmodel()
    result = model.transcribe(file_path)
    audio_data = []
    for segment in result["segments"]:
        item = {
            "start": segment["start"],
            "end": segment["end"],
            "text": "Text: " + segment["text"],
            "file_id": unique_key,
        }
        audio_data.append(item)

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
