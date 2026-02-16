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


def get_vision_data(video_path):
    processor, model, device = Loadmodel()
    visual_data = []
    capture = cv.VideoCapture(video_path)

    fps = int(capture.get(cv.CAP_PROP_FPS))
    totalfps = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    if fps == 0 or totalfps == 0:
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
        output = model.generate(**input)
        item = {
            "start": i / fps,
            "text": "Visual: " + processor.decode(output[0], skip_special_tokens=True),
        }
        visual_data.append(item)

    capture.release()
    return visual_data
