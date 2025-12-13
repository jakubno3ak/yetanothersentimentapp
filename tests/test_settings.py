from settings import Settings


class TestSettings:
    def test_settings_default_values(self):
        settings = Settings()
        assert settings.model_dir == "model"
        assert settings.sentence_transformer_dir == "model/sentence_transformer.model"
        assert settings.classifier_joblib_path == "model/classifier.joblib"
        assert settings.onnx_embedding_model_path == "model/model.onnx"
        assert settings.onnx_classifier_path == "model/classifier.onnx"
        assert settings.onnx_tokenizer_path == "model/tokenizer.json"
        assert settings.embedding_dim == 384

    def test_settings_custom_bucket(self):
        settings = Settings(s3_bucket_name="custom-bucket")
        assert settings.s3_bucket_name == "custom-bucket"
