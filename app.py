import joblib
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI
from tokenizers import Tokenizer

from src.app.models import PredictRequest, PredictResponse

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}

app = FastAPI()

try:
    tokenizer = Tokenizer.from_file("model/tokenizer.json")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

try:
    session = ort.InferenceSession("model/model.onnx")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

try:
    classifier = joblib.load("model/classifier.joblib")
except Exception as e:
    print(f"Error loading classifier: {e}")
    classifier = None


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not tokenizer or not session or not classifier:
        return PredictResponse(prediction="error: model not loaded")

    # tokenize input
    encoded = tokenizer.encode(request.text)

    # numpy arrays for ONNX
    input_ids = np.array([encoded.ids])
    attention_mask = np.array([encoded.attention_mask])

    # run inference
    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    outputs = session.run(None, onnx_inputs)[0]
    prediction = classifier.predict(outputs)[0]
    label = SENTIMENT_MAP.get(prediction, "unknown")
    return PredictResponse(prediction=label)