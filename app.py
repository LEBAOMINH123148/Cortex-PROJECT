import gradio as gr
import os
import shutil
from The_Brain import Working

def process_file(uploaded_file, v_query, a_query):
    if uploaded_file is None:
        return "Please upload a file.", None, None

    original_filename = os.path.basename(uploaded_file.name)
    file_ext = os.path.splitext(original_filename)[1].lower()
    
    # We copy the file with its original extension so cv2 and ffmpeg can read it correctly
    temp_path = "temp_uploaded_file" + file_ext
    shutil.copy(uploaded_file.name, temp_path)
    
    file_size = os.path.getsize(temp_path)
    unique_key = f"{original_filename}_{file_size}"
    
    if not (v_query or a_query):
        return "Please tell us what you want to find first.", None, None
        
    try:
        if file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
            result_md = Working(v_query, a_query, temp_path, unique_key, original_filename)
            return result_md, temp_path, None
        else:
            result_md = Working("", a_query, temp_path, unique_key, original_filename)
            return result_md, None, temp_path
    except Exception as e:
        return f"Error: {str(e)}", None, None

with gr.Blocks(title="Cortex") as app:
    gr.Markdown("<h1 style='text-align: center;'><span style='color: red;'>Welcome to Cortex</span></h1>")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload your file here (Video or Audio)")
            v_query_input = gr.Textbox(label="What do you want to find? (Visual query for video)")
            a_query_input = gr.Textbox(label="What do you want to find? (Audio query)")
            submit_btn = gr.Button("Search", variant="primary")
            
        with gr.Column():
            output_text = gr.Markdown(label="Results")
            output_video = gr.Video(label="Video Player")
            output_audio = gr.Audio(label="Audio Player")

    submit_btn.click(
        fn=process_file,
        inputs=[file_input, v_query_input, a_query_input],
        outputs=[output_text, output_video, output_audio]
    )

if __name__ == "__main__":
    app.launch(share=False)
