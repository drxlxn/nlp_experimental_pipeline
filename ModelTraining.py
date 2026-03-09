import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Scikit-Learn Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, learning_curve, train_test_split

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. The Blueprint (Strategy Interface)
# ==========================================
class ModelStrategy:
    """Base interface for all machine learning models."""

    def __init__(self):
        self.model = None
        self.param_grid = {}

    def train(self, X_train, y_train):
        print(f"Training {self.__class__.__name__} with default parameters...")
        self.model.fit(X_train, y_train)

    def tune_hyperparameters(self, X_train, y_train, n_iter: int = 5):
        print(f"Starting Randomized Hyperparameter Tuning for {self.__class__.__name__}...")
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            n_iter=n_iter,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        search.fit(X_train, y_train)
        print(f"Best Parameters Found: {search.best_params_}")
        self.model = search.best_estimator_
        return search.best_params_

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred) -> Dict[str, float]:
        """Evaluates models using standard classification metrics."""
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

    def plot_learning_curve(self, X, y, output_dir: str, title: str):
        """Generates and saves a learning curve visualization using cross-validation."""
        print("Generating learning curve...")
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=3, scoring='f1', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker='o')
        plt.plot(train_sizes, test_mean, label="Cross-Validation Score", color="orange", marker='o')
        plt.title(f"Learning Curve: {title}")
        plt.xlabel("Training Set Size")
        plt.ylabel("F1 Score")
        plt.legend(loc="best")
        plt.grid()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_learning_curve.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Learning curve saved to {save_path}")

    def save_model(self, filepath: str):
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


# ==========================================
# 2. Scikit-Learn Concrete Models
# ==========================================
class LogisticRegressionModel(ModelStrategy):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }


class SVMModel(ModelStrategy):
    def __init__(self):
        super().__init__()
        self.model = SVC(kernel='linear', random_state=42)
        self.param_grid = {
            'C': [0.1, 1.0, 10.0]
            # Keeping strictly linear. RBF kernel on 5000 features will freeze standard machines.
        }


# ==========================================
# 3. PyTorch Deep Learning Architecture
# ==========================================
class SimpleTextDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(SimpleTextDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)


class PyTorchDNNModel(ModelStrategy):
    """Overrides the base strategy to handle PyTorch's custom training requirements."""

    def __init__(self, device_preference: str = "auto"):
        super().__init__()
        self.model = None
        self.epochs = 10
        self.batch_size = 64
        self.lr = 0.001
        self.training_losses = []

        # CPU vs GPU Toggle Setup
        if device_preference == "gpu" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device_preference == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_preference == "cpu":
            self.device = torch.device("cpu")
        else:
            # Auto mode
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        print(f"PyTorch initialized. Using device: {self.device}")

    def train(self, X_train, y_train):
        print(f"Training PyTorch DNN for {self.epochs} epochs...")

        # 1. Initialize network dynamically based on feature matrix width
        input_dim = X_train.shape[1]
        self.model = SimpleTextDNN(input_dim=input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 2. Prepare DataLoaders
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 3. Training Loop
        self.model.train()
        self.training_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.training_losses.append(avg_loss)
            print(f"  -> Epoch {epoch + 1}/{self.epochs} | Loss: {avg_loss:.4f}")

    def tune_hyperparameters(self, X_train, y_train, n_iter: int = 5):
        print("Starting custom hyperparameter tuning for PyTorch DNN...")
        # Since sklearn RandomizedSearchCV doesn't work with PyTorch natively, we do a basic search
        learning_rates = [0.001, 0.01]
        batch_sizes = [32, 64, 128]

        best_f1 = 0
        best_params = {}

        # Create a quick validation split for tuning
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Test a few random combinations
        for _ in range(n_iter):
            lr = np.random.choice(learning_rates)
            bs = int(np.random.choice(batch_sizes))

            self.lr = lr
            self.batch_size = bs
            print(f"Testing Config - LR: {lr}, Batch Size: {bs}")

            self.train(X_t, y_t)
            preds = self.predict(X_v)
            score = f1_score(y_v, preds, zero_division=0)

            if score > best_f1:
                best_f1 = score
                best_params = {'learning_rate': lr, 'batch_size': bs}

        print(f"Best Parameters Found: {best_params} with Validation F1: {best_f1:.4f}")
        self.lr = best_params['learning_rate']
        self.batch_size = best_params['batch_size']

        # Retrain on full dataset with best params
        self.train(X_train, y_train)
        return best_params

    def predict(self, X_test):
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs >= 0.5).float().cpu().numpy().squeeze()
        return predictions

    def plot_learning_curve(self, X, y, output_dir: str, title: str):
        """Overrides the standard curve to plot Deep Learning Epoch Loss instead."""
        print("Generating Deep Learning training curve...")
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.training_losses) + 1), self.training_losses, marker='o', color='red')
        plt.title(f"Training Loss Curve: {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross Entropy Loss")
        plt.grid()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_loss_curve.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Loss curve saved to {save_path}")

    def save_model(self, filepath: str):
        # Change the extension for PyTorch
        filepath = filepath.replace(".pkl", ".pth")
        torch.save(self.model.state_dict(), filepath)
        print(f"PyTorch model weights saved to {filepath}")

    def load_model(self, filepath: str):
        filepath = filepath.replace(".pkl", ".pth")
        # Note: Model must be initialized with input_dim before loading weights
        if self.model is None:
            raise RuntimeError("You must initialize the PyTorch model with data before loading weights.")
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"PyTorch model loaded from {filepath}")


# ==========================================
# 4. The Trainer Class (Orchestrator)
# ==========================================
class ModelTrainer:
    def __init__(self, model_strategy: ModelStrategy):
        self.strategy = model_strategy

    def run_training(self, X_train, y_train, X_test, y_test, do_tune: bool = False, model_save_path: str = None,
                     plot_dir: str = None, experiment_name: str = "Model"):

        best_params = None
        if do_tune:
            best_params = self.strategy.tune_hyperparameters(X_train, y_train)
        else:
            self.strategy.train(X_train, y_train)

        if plot_dir:
            self.strategy.plot_learning_curve(X_train, y_train, output_dir=plot_dir, title=experiment_name)

        predictions = self.strategy.predict(X_test)

        # Fulfills requirement: evaluating using accuracy, precision, recall, or F1 score
        metrics = self.strategy.evaluate(y_test, predictions)

        if model_save_path:
            self.strategy.save_model(model_save_path)

        return metrics, best_params