from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_bucket_name: str = "mlops-lab11-models-kubanowak"
    
    model_dir: str = "model"

    sentence_transformer_dir: str = "model/sentence_transformer.model"
    classifier_joblib_path: str = "model/classifier.joblib"
    
    onnx_embedding_model_path: str = "model/model.onnx"
    onnx_classifier_path: str = "model/classifier.onnx"
    onnx_tokenizer_path: str = "model/tokenizer.json"
    embedding_dim: int = 384


settings = Settings()
