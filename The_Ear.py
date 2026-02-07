import whisper
import json

# use model
model = whisper.load_model("small.en")
result = model.transcribe("Chàng_Trai_Tự_Tin_Trước_Lớp.mp4")
print(result["text"])

# create data to save in json file
data_to_save = []
for segment in result["segments"]:
    item = {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
    data_to_save.append(item)

with open("transcript_file.jason", "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, indent=4, ensure_ascii=False)
