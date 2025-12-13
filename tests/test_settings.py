import pytest

from settings import Settings


class TestSettings:
    def test_settings_default_values(self):
        settings = Settings()
        assert settings.s3_bucket_name == "mlops-lab11-models-kubano"
        assert settings.model_path == "model/sentence_transformer.model"
        assert settings.classifier_path == "model/classifier.joblib"
        assert settings.onnx_model_path == "model"
        assert settings.onnx_model_name == "model.onnx"

    def test_settings_tokenizer_path(self):
        settings = Settings()
        assert settings.tokenizer_path == "model/sentence_transformer.model"

    def test_settings_custom_bucket(self):
        settings = Settings(s3_bucket_name="custom-bucket")
        assert settings.s3_bucket_name == "custom-bucket"
