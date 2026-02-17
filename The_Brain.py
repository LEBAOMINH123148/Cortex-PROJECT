import streamlit as st
from sentence_transformers import SentenceTransformer, util
from The_Eye import get_vision_data
from The_Ear import get_audio_data


@st.cache_resource
def Loadmodel():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# The Ear and The Eye, return audio and vision after transcribed
@st.cache_data
def process_uploadedfile(file_path, unique_key):
    user_data = []
    audio_data = get_audio_data(file_path)
    vision_data = get_vision_data(file_path)
    user_data = vision_data + audio_data
    user_data.sort(
        key=lambda x: x["start"]
    )  # sort by start time respectively based on the order of each variable in the line above

    model = Loadmodel()
    sentences_to_check = []
    for segment in user_data:
        sentences_to_check.append(segment["text"])  # embedd the user_data into vector
    embedding = model.encode(sentences_to_check)

    return embedding, user_data, model


# The Brain
def Working(user_input, file_path, unique_key):
    embedding, user_data, model = process_uploadedfile(file_path, unique_key)
    user_input = model.encode(user_input.lower())  # embedd the user_input into vector
    result = util.semantic_search(user_input, embedding, top_k=3)  # compare them
    for i in result[0]:
        id = i["corpus_id"]
        score = i["score"] * 100
        text = user_data[id]["text"]
        time_start = user_data[id]["start"]
        st.write(f"Time start: {time_start:.1f}s {text}. Percentage: {score:.1f}%\n")
        st.audio(file_path, format="audio/wav", start_time=time_start)
