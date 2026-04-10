import gradio as gr
from transformers import pipeline

pipe = pipeline("audio-classification", model="afloven/messymashupclassifier")

def classify_audio(audio):
    preds = pipe(audio)
    return {p["label"]: p["score"] for p in preds}

demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Label(num_top_classes=5),
    title="Messy Mashup Genre Classifier",
    description="Upload a 10-second audio clip to see which genre it belongs to."
)

demo.launch()
