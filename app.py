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

    with st.form(key="search_form"):
        query = st.text_input("What do you want to find?: ")
        submit_button = st.form_submit_button("Search")

        if submit_button:
            if query:
                unique_key = f"{uploaded_file.name}_{uploaded_file.size}"
                Working(query, "temp_uploaded_file", unique_key)
            else:
                st.warning("Please tell us what you want to find first")

    os.remove("temp_uploaded_file")
