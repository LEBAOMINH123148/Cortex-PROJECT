import cv2 as cv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
import torch


@st.cache_resource
def Loadmodel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model, device


def get_vision_data(video_path, unique_key):
    processor, model, device = Loadmodel()
    visual_data = []
    capture = cv.VideoCapture(video_path)

    try:
        fps = int(capture.get(cv.CAP_PROP_FPS))
        totalfps = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or totalfps <= 0:  # file type check
            print("File is mp3 or error")
            capture.release()
            return []
    except ValueError:  # if fps = int(NaN)
        capture.release()
        return []

    for i in range(0, totalfps, fps * 5):
        capture.set(cv.CAP_PROP_POS_FRAMES, i)
        sucess, frame = capture.read()
        if not sucess:
            break
        framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # change from bgr to rgb
        pil_image = Image.fromarray(framergb)  # create the image memory
        input = processor(pil_image, return_tensors="pt").to(device)
        output = model.generate(
            **input,
            repetition_penalty=1.5,
        )
        item = {
            "start": max(0, (i / fps) - 1),
            "end": min((i / fps) + 5, totalfps / fps),
            "text": "Visual: " + processor.decode(output[0], skip_special_tokens=True),
            "file_id": unique_key,
        }
        visual_data.append(item)

    capture.release()
    # stuff for Vcollection
    Vlist_ids = []
    Vlist_document = []
    Vlist_metadatas = []
    n = 1
    for i in visual_data:
        Vlist_ids.append(f"{unique_key}_{n}")
        Vlist_document.append(i["text"])
        item = {"start": i["start"], "end": i["end"], "file_id": i["file_id"]}
        Vlist_metadatas.append(item)
        n += 1
    return Vlist_ids, Vlist_document, Vlist_metadatas
