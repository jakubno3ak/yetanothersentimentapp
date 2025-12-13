import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .text_processor import TextProcessor


SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentModel:
    
    def __init__(
        self,
        tokenizer_path: str,
        embedding_model_path: str,
        classifier_path: str
    ):
        self.text_processor = TextProcessor()
        self.tokenizer = None
        self.embedding_session = None
        self.classifier_session = None
        
        self._load_tokenizer(tokenizer_path)
        self._load_embedding_model(embedding_model_path)
        self._load_classifier(classifier_path)
    
    def _load_tokenizer(self, path: str) -> None:
        try:
            self.tokenizer = Tokenizer.from_file(path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
    
    def _load_embedding_model(self, path: str) -> None:
        try:
            self.embedding_session = ort.InferenceSession(path)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
    
    def _load_classifier(self, path: str) -> None:
        try:
            self.classifier_session = ort.InferenceSession(path)
        except Exception as e:
            print(f"Error loading classifier: {e}")
    
    def is_ready(self) -> bool:
        return all([self.tokenizer, self.embedding_session, self.classifier_session])
    
    def predict(self, text: str) -> str:
        if not self.is_ready():
            return "error: model not loaded"
        
        # clean and normalize text
        cleaned_text = self.text_processor.clean_text(text)
        
        # tokenize input
        encoded = self.tokenizer.encode(cleaned_text)
        
        # prepare numpy arrays for ONNX
        input_ids = np.array([encoded.ids])
        attention_mask = np.array([encoded.attention_mask])
        
        # run embedding inference
        embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        embeddings = self.embedding_session.run(None, embedding_inputs)[0]
        
        # run classifier inference
        classifier_input_name = self.classifier_session.get_inputs()[0].name
        classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
        prediction = self.classifier_session.run(None, classifier_inputs)[0]
        
        label = SENTIMENT_MAP.get(prediction[0], "unknown")
        
        return label
