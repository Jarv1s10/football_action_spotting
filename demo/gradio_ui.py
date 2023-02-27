import os
import gradio as gr
from model_inference import inference_video

demo = gr.Interface(inference_video, 'video', 'playable_video',
                    examples=[os.path.join(os.path.dirname(__file__), 'gradio_cached_examples', '12', 'input', 'example.mp4')],
                    cache_examples=True)


if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)