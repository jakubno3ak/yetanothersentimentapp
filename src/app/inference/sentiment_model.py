import joblib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .text_processor import TextProcessor


SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentModel:
    def __init__(self, tokenizer_path: str, onnx_model_path: str, classifier_path: str):
        self.text_processor = TextProcessor()
        self.tokenizer = None
        self.session = None
        self.classifier = None
        
        self._load_tokenizer(tokenizer_path)
        self._load_onnx_model(onnx_model_path)
        self._load_classifier(classifier_path)
    
    def _load_tokenizer(self, path: str) -> None:
        try:
            self.tokenizer = Tokenizer.from_file(path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
    
    def _load_onnx_model(self, path: str) -> None:
        try:
            self.session = ort.InferenceSession(path)
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
    
    def _load_classifier(self, path: str) -> None:
        try:
            self.classifier = joblib.load(path)
        except Exception as e:
            print(f"Error loading classifier: {e}")
    
    def is_ready(self) -> bool:
        return all([self.tokenizer, self.session, self.classifier])
    
    def predict(self, text: str) -> str:
        if not self.is_ready():
            return "error: model not loaded"
        
        # clean and normalize text
        cleaned_text = self.text_processor.clean_text(text)
        
        # tokenize input
        encoded = self.tokenizer.encode(cleaned_text)
        
        # trepare numpy arrays for ONNX
        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])
        
        # run inference
        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.session.run(None, onnx_inputs)[0]
        
        # classify
        prediction = self.classifier.predict(outputs)[0]
        label = SENTIMENT_MAP.get(prediction, "unknown")
        
        return label
