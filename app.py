import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util

st.title("OHaiYo :red[ONi-chaan]", text_alignment="center")


@st.cache_resource
def initModel():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# load data and embedd it
@st.cache_data
def Loaddata(_model):
    with open("transcript_file.jason") as f:
        data = json.load(f)
    sentences_to_check = []
    for segment in data:
        sentences_to_check.append(segment["text"])
    embedding = model.encode(sentences_to_check)
    return data, embedding


def Working(user_input, data, model, embedding):
    user_input = model.encode(user_input.lower())
    result = util.semantic_search(user_input, embedding, top_k=3)
    for i in result[0]:
        id = i["corpus_id"]
        score = i["score"] * 100
        text = data[id]["text"]
        time_start = data[id]["start"]
        st.write(
            f"Time: {time_start:.1f}s, Sentence {id}: '{text}' Percentage: {score:.1f}%\n"
        )
        st.audio("Recording.m4a", format="audio/wav", start_time=time_start)


model = initModel()
data, embedding = Loaddata(model)

st.text_input("What do you want to find?: ", key="user_input")
if st.session_state.user_input:  # avoid showing result before input
    Working(st.session_state.user_input, data, model, embedding)
