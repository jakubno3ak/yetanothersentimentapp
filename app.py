from fastapi import FastAPI

from src.app.models import PredictRequest, PredictResponse
from src.app.inference.sentiment_model import SentimentModel

app = FastAPI()

model = SentimentModel(
    tokenizer_path="model/tokenizer.json",
    onnx_model_path="model/model.onnx",
    classifier_path="model/classifier.joblib",
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    label = model.predict(request.text)
    return PredictResponse(prediction=label)
