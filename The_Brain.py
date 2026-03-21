import streamlit as st
from The_Eye import get_vision_data
from The_Ear import get_audio_data
import chromadb

client = chromadb.PersistentClient(path="./cortexdb")
Acollection = client.get_or_create_collection(name="Audio")
Vcollection = client.get_or_create_collection(name="Video")


# The Ear and The Eye, return audio and vision after transcribed to put it in chroma collection
@st.cache_data
def process_uploadedfile(file_path, unique_key, filename):
    Alist_ids, Alist_document, Alist_metadatas = get_audio_data(file_path, unique_key)
    if len(Alist_ids) > 0:  # for no audio in mp3
        Acollection.upsert(
            ids=Alist_ids, documents=Alist_document, metadatas=Alist_metadatas
        )

    if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        Vlist_ids, Vlist_document, Vlist_metadatas = get_vision_data(
            file_path, unique_key
        )
        if len(Vlist_ids) > 0:  # for no recognizeable frame in video
            Vcollection.upsert(
                ids=Vlist_ids, documents=Vlist_document, metadatas=Vlist_metadatas
            )


# The Brain
def Working(Vuser_input, Auser_input, file_path, unique_key, filename):
    process_uploadedfile(file_path, unique_key, filename)
    Aresult = Acollection.query(
        where={"file_id": unique_key},
        query_texts=[Auser_input],
        n_results=3,
    )
    Vresult = Vcollection.query(
        where={"file_id": unique_key},
        query_texts=[Vuser_input],
        n_results=3,
    )

    Ameta = Aresult["metadatas"][0]
    Adoc = Aresult["documents"][0]
    Adis = Aresult["distances"][0]
    if filename.lower().endswith(
        (".mp4", ".mov", ".avi", ".mkv")
    ):  # for video, merge audio and vision result based on time and show the video with matched time
        Vmeta = Vresult["metadatas"][0]
        Vdoc = Vresult["documents"][0]
        Vdis = Vresult["distances"][0]

        merged = []
        for i in range(len(Ameta)):
            A_start = Ameta[i]["start"]
            A_end = Ameta[i]["end"]
            Atext = Adoc[i]
            if Adis[i] < 1:
                Ascore = abs((1 - Adis[i]) * 100)
            else:
                Ascore = 0
            for j in range(len(Vmeta)):
                V_start = Vmeta[j]["start"]
                V_end = Vmeta[j]["end"]
                Vtext = Vdoc[j]
                if A_start <= V_end and A_end >= V_start:
                    if Vdis[j] < 1:
                        Vscore = abs((1 - Vdis[j]) * 100)
                    else:
                        Vscore = 0
                    combined_score = (Ascore + Vscore) / 2
                    merged.append((A_start, A_end, Atext, Vtext, combined_score))

        merged.sort(key=lambda x: x[4], reverse=True)  # sort by score
        if merged:
            for item in merged:
                time_start = item[0]
                text = item[2] + " " + item[3]
                score = item[4]
                st.write(
                    f"Time start: {time_start:.1f}s {text} - Percentage: {score:.1f}%\n"
                )
                st.video(file_path, format="video/mp4", start_time=time_start)
    else:
        for i in range(len(Ameta)):
            time_start = Ameta[i]["start"]
            text = Adoc[i]
            if Adis[i] < 1:
                score = abs((1 - Adis[i]) * 100)
            else:
                score = 0

            st.write(
                f"Time start: {time_start:.1f}s {text} - Percentage: {score:.1f}%\n"
            )
            st.audio(file_path, format="audio/wav", start_time=time_start)
