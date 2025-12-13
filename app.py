from mangum import Mangum
from fastapi import FastAPI

from src.app.models import PredictRequest, PredictResponse
from src.app.inference.sentiment_model import SentimentModel
from settings import settings

app = FastAPI()

model = SentimentModel(
    tokenizer_path=settings.onnx_tokenizer_path,
    embedding_model_path=settings.onnx_embedding_model_path,
    classifier_path=settings.onnx_classifier_path,
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    label = model.predict(request.text)
    return PredictResponse(prediction=label)

handler = Mangum(app)