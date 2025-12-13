import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from settings import Settings


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.classifier_path}...")
    classifier = joblib.load(settings.classifier_path)

    # fefine input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_type,
        target_opset=12,
    )

    print(f"Saving ONNX model to {settings.onnx_classifier_path}...")
    with open(settings.onnx_classifier_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
