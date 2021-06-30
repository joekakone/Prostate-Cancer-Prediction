#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Restore model
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


inputs = [
    gr.inputs.Number(label="Radius"),
    gr.inputs.Slider(minimum=0, maximum=100, step=None, label="Texture"),
    gr.inputs.Number(label="Perimeter"),
    gr.inputs.Number(label="Area"),
    gr.inputs.Number(label="Smoothness"),
    gr.inputs.Number(label="Compactness"),
    gr.inputs.Slider(minimum=0, maximum=100, step=None, label="Symmetry"),
    gr.inputs.Number(label="Fractal dimension")
]

output = gr.outputs.Label(label="Diagnosis")

# Serving Function
def serving(radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension):
    data = {
        "radius": [radius],
        "texture": [texture],
        "perimeter": [perimeter],
        "area": [area],
        "smoothness": [smoothness],
        "compactness": [compactness],
        "symmetry": [symmetry],
        "fractal_dimension": [fractal_dimension]
    }
    data = pd.DataFrame(data)
    data = scaler.transform(data)
    output = model.predict_proba(data)[0]
    output = {encoder.inverse_transform([i])[0]: proba for i, proba in enumerate(output)}

    return output


# Interface
interface = gr.Interface(
    fn=serving,
    inputs=inputs,
    outputs=output,
    title="Cancer Diagnosis",
    # server_port=7869,
    server_name="0.0.0.0",
    verbose=True,
)

if __name__ == "__main__":
    interface.launch()
