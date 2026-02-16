import cv2 as cv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st


@st.cache_resource
def Loadmodel():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cuda")
    return processor, model


def get_vision_data(video_path):
    processor, model = Loadmodel()
    visual_data = []
    capture = cv.VideoCapture(video_path)

    fps = int(capture.get(cv.CAP_PROP_FPS))
    totalfps = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    for i in range(0, totalfps, fps * 2):
        capture.set(cv.CAP_PROP_POS_FRAMES, i)
        sucess, frame = capture.read()
        if not sucess:
            break
        framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # change from bgr to rgb
        pil_image = Image.fromarray(framergb)  # create the image memory
        input = processor(pil_image, return_tensors="pt").to("cuda")
        output = model.generate(**input)
        item = {
            "start": i / fps,
            "text": "Visual: " + processor.decode(output[0], skip_special_tokens=True),
        }
        visual_data.append(item)

    capture.release()
    return visual_data
