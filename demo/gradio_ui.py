import os
import gradio as gr
from model_inference import inference_video, get_predictions, render_predictions_on_video

# demo = gr.Interface(inference_video, 'video', 'playable_video',
#                     cache_examples=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():    
            video_input = gr.Video(label='Video')
            with gr.Row():
                submit_btn = gr.Button("Get JSON")
                preds_on_video = gr.Button("Play video with predictions")
        with gr.Column():
            json_output = gr.JSON(label='Json output')
    
    submit_btn.click(fn=get_predictions, inputs=video_input, outputs=json_output)
    preds_on_video.click(fn=render_predictions_on_video, inputs=[video_input, json_output])

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)