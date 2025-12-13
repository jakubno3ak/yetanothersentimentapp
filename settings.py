from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_bucket_name: str = "mlops-lab11-models-kubano"
    model_path: str = "model/sentence_transformer.model"
    tokenizer_path: str = "model/sentence_transformer.model"
    onnx_model_path: str = "model"
    onnx_model_name: str = "model.onnx"
    classifier_path: str = "model/classifier.joblib"
    onnx_classifier_path: str = "model/classifier.onnx"
    embedding_dim: int = 384


settings = Settings()
