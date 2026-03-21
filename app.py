import streamlit as st
import os
from The_Brain import Working

st.title("OHaiYo :red[ONi-chaan]", text_alignment="center")

uploaded_file = st.file_uploader("Upload your file here(Video or audio): ")
if uploaded_file:
    with open("temp_uploaded_file", "wb") as f:
        f.write(
            uploaded_file.getvalue()  # create temp file that has content of the user uploaded file to put it in whisper
        )  # whisper need file path not file on ram (st.file_upload store data on ram)
    st.success("File Uploaded")

    if uploaded_file.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        with st.form(key="search_form"):
            Vquery = st.text_input("What do you want to find?(visual): ")
            Aquery = st.text_input("What do you want to find?(audio): ")
            submit_button = st.form_submit_button("Search")
            if submit_button:
                if Vquery or Aquery:
                    unique_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    filename = uploaded_file.name
                    Working(Vquery, Aquery, "temp_uploaded_file", unique_key, filename)
                else:
                    st.warning("Please tell us what you want to find first")
    else:
        with st.form(key="search_form"):
            Aquery = st.text_input("What do you want to find?: ")
            submit_button = st.form_submit_button("Search")
            if submit_button:
                if Aquery:
                    unique_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    filename = uploaded_file.name
                    Working("", Aquery, "temp_uploaded_file", unique_key, filename)
                else:
                    st.warning("Please tell us what you want to find first")

    os.remove("temp_uploaded_file")
