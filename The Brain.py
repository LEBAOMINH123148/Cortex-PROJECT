import json
from sentence_transformers import SentenceTransformer, util

# load data from json file
with open("transcript_file.jason") as f:
    if f is None:
        print("No transcript file")
        exit()
    else:
        data = json.load(f)

# load model to traform sentence to vector
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences_to_check = []
for segment in data:
    sentences_to_check.append(segment["text"])
embedding = model.encode(sentences_to_check)


while True:
    print("What do you want to find?(type q to exit): ")
    x = input().lower()
    if x == "q":
        break

    user_input = model.encode(x)
    result = util.semantic_search(user_input, embedding, top_k=3)
    for i in result[0]:
        id = i["corpus_id"]
        score = i["score"] * 100
        text = data[id]["text"]
        time_start = data[id]["start"]
        print(f"Time: {time_start}s, Sentence {id}:'{text}' Percentage:{score:.1f}%")
    print("-" * 20)
