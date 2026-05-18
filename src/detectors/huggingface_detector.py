from transformers import pipeline
from src.data.preprocessor import clean_text

#Fine-tuned model specifically for phishing detection
MODEL_NAME = "ealvaradob/bert-finetuned-phishing"
MAX_LENGTH = 512


class HuggingFaceDetector:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"[HuggingFace] Loading model: {model_name} ...")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        print("[HuggingFace] Model loaded.")