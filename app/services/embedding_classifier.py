from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "embedding_model"

CLASSIFIER_PATH = MODEL_DIR / "embedding_classifier.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "metadata.pkl"


class EmbeddingModelNotLoadedError(Exception):
    pass


class ResumeEmbeddingClassifier:
    def __init__(self) -> None:
        self.embedding_model: Optional[SentenceTransformer] = None
        self.classifier: Optional[Any] = None
        self.label_encoder: Optional[Any] = None
        self.metadata: Optional[dict[str, Any]] = None

        self.load()

    def load(self) -> None:
        if not CLASSIFIER_PATH.exists():
            raise FileNotFoundError(f"Classifier file not found: {CLASSIFIER_PATH}")

        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

        self.classifier = joblib.load(CLASSIFIER_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.metadata = joblib.load(METADATA_PATH)

        embedding_model_name = self.metadata.get("embedding_model_name")

        if not embedding_model_name:
            raise EmbeddingModelNotLoadedError(
                "Embedding model name is missing in metadata"
            )

        self.embedding_model = SentenceTransformer(embedding_model_name)

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        if (
            self.embedding_model is None
            or self.classifier is None
            or self.label_encoder is None
            or self.metadata is None
        ):
            raise EmbeddingModelNotLoadedError("Embedding classifier is not loaded")

        if not text or not text.strip():
            raise ValueError("Input text is empty")

        embedding = self.embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        predicted_encoded = self.classifier.predict(embedding)[0]
        predicted_category = self.label_encoder.inverse_transform(
            [predicted_encoded]
        )[0]

        if hasattr(self.classifier, "predict_proba"):
            scores = self.classifier.predict_proba(embedding)[0]
            encoded_classes = self.classifier.classes_
        elif hasattr(self.classifier, "decision_function"):
            scores = self._scores_to_probabilities(
                self.classifier.decision_function(embedding)
            )
            encoded_classes = np.arange(len(scores))
        else:
            scores = np.zeros(len(self.label_encoder.classes_))
            scores[int(predicted_encoded)] = 1.0
            encoded_classes = np.arange(len(scores))

        top_indices = np.argsort(scores)[::-1][:top_k]

        top_categories = []

        for index in top_indices:
            encoded_class = int(encoded_classes[index])
            category = self.label_encoder.inverse_transform([encoded_class])[0]

            top_categories.append(
                {
                    "category": str(category),
                    "score": float(scores[index]),
                }
            )

        return {
            "predicted_category": str(predicted_category),
            "confidence": float(max(item["score"] for item in top_categories)),
            "top_3_categories": top_categories,
            "model_type": "embedding",
            "embedding_model": self.metadata.get("embedding_model_name"),
            "classifier_name": self.metadata.get("classifier_name"),
        }

    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)

        if scores.ndim > 1:
            scores = scores[0]

        if scores.ndim == 1 and len(scores) == 1:
            positive_score = 1 / (1 + np.exp(-scores[0]))
            return np.array([1 - positive_score, positive_score])

        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)

        return exp_scores / exp_scores.sum()


resume_embedding_classifier = ResumeEmbeddingClassifier()
