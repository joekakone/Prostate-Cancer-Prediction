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
    gr.inputs.Slider(minimum=9, maximum=25, step=None, label="Radius"),
    gr.inputs.Slider(minimum=11, maximum=27, step=None, label="Texture"),
    gr.inputs.Slider(minimum=52, maximum=172, step=None, label="Perimeter"),
    gr.inputs.Slider(minimum=202, maximum=1878, step=None, label="Area"),
    gr.inputs.Slider(minimum=0.070, maximum=0.143, step=None, label="Smoothness"),
    gr.inputs.Slider(minimum=0.038, maximum=0.345, step=None, label="Compactness"),
    gr.inputs.Slider(minimum=0.135, maximum=0.304, step=None, label="Symmetry"),
    gr.inputs.Slider(minimum=0.053, maximum=0.097, step=0.001, label="Fractal dimension"),
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
