import os
import joblib
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# ==========================================
# 1. The Blueprint (Strategy Interface)
# ==========================================
class ModelStrategy:
    """Base interface for all machine learning models."""

    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        print(f"Training {self.__class__.__name__}...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred) -> Dict[str, float]:
        """Calculates standard classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }

        print(f"\n--- {self.__class__.__name__} Results ---")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        return metrics

    def save_model(self, filepath: str):
        """Saves the trained model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Loads a trained model from disk."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


# ==========================================
# 2. The Concrete Models (The Lego Blocks)
# ==========================================
class LogisticRegressionModel(ModelStrategy):
    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        super().__init__()
        # C is the inverse of regularization strength (smaller values specify stronger regularization)
        self.model = LogisticRegression(max_iter=max_iter, C=C, random_state=42)


class RandomForestModel(ModelStrategy):
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Uses all available CPU cores!
        )


class NaiveBayesModel(ModelStrategy):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        # Standard baseline for text classification (especially Bag of Words/TF-IDF)
        self.model = MultinomialNB(alpha=alpha)


# ==========================================
# 3. The Trainer Class (Orchestrator)
# ==========================================
class ModelTrainer:
    """Manages the training, evaluation, and saving process."""

    def __init__(self, model_strategy: ModelStrategy):
        self.strategy = model_strategy
        self.metrics = None

    def run_training(self, X_train, y_train, X_test, y_test, save_path: str = None):
        # 1. Train
        self.strategy.train(X_train, y_train)

        # 2. Predict
        predictions = self.strategy.predict(X_test)

        # 3. Evaluate
        self.metrics = self.strategy.evaluate(y_test, predictions)

        # 4. Save
        if save_path:
            self.strategy.save_model(save_path)

        return self.metrics