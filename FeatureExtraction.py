from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec


# =========================
# 1. The Blueprints
# =========================

class FeatureExtractionStrategy:
    def fit_transform(self, train_texts: List[str]) -> np.ndarray:
        raise NotImplementedError()

    def transform(self, test_texts: List[str]) -> np.ndarray:
        raise NotImplementedError()

    def get_feature_names(self) -> List[str]:
        raise NotImplementedError()


# =========================
# 2. Helper
# =========================

def identity_tokenizer(text):
    return text

# =========================
# 3. Feature Extractors
# =========================

# 1. Create a Base class for Scikit-Learn vectorizers
class BaseSklearnExtractor(FeatureExtractionStrategy):
    # This assumes self.vectorizer is defined in the child class

    def fit_transform(self, train_tokens: List[List[str]]) -> np.ndarray:
        return self.vectorizer.fit_transform(train_tokens).toarray()

    def transform(self, test_tokens: List[List[str]]) -> np.ndarray:
        return self.vectorizer.transform(test_tokens).toarray()

    def get_feature_names(self) -> List[str]:
        return [name for name in self.vectorizer.get_feature_names_out()]


# 2. Inherit from the Base class and only write the setup logic
class BagOfWordsExtractor(BaseSklearnExtractor):
    def __init__(self, max_features: int = 5000):
        self.vectorizer = CountVectorizer(
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            max_features=max_features
        )


class TfidfExtractor(BaseSklearnExtractor):
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=identity_tokenizer,
            preprocessor=identity_tokenizer,
            token_pattern=None,
            max_features=max_features
        )


class Word2VecExtractor(FeatureExtractionStrategy):
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4, sg: int = 0):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None

    def _document_vector(self, tokens: List[str]) -> np.ndarray:
        # Averages the word vectors to create a single vector for the entire comment
        valid_tokens = [token for token in tokens if token in self.model.wv]
        if not valid_tokens:
            return np.zeros(self.vector_size)
        return np.mean([self.model.wv[token] for token in valid_tokens], axis=0)

    def fit_transform(self, train_tokens: List[List[str]]) -> np.ndarray:
        # Trains the Gensim model directly on your lists of words
        self.model = Word2Vec(
            sentences=train_tokens,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        return np.array([self._document_vector(tokens) for tokens in train_tokens])

    def transform(self, test_tokens: List[List[str]]) -> np.ndarray:
        return np.array([self._document_vector(tokens) for tokens in test_tokens])

    def get_feature_names(self) -> List[str]:
        # Word2Vec features don't have human-readable names, just indices 0 to vector_size
        return [str(i) for i in range(self.vector_size)]


# =========================
# 4. Main Pipeline
# =========================

class FeatureExtractionPipeline:
    def __init__(self, extractor: FeatureExtractionStrategy, base_path: str = "data"):
        self.extractor = extractor
        self.base_path = Path(__file__).resolve().parent / base_path

    def load_data(
        self,
        train_file: str = "train_binary_labels.csv",
        test_file: str = "test_binary_labels.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.base_path / train_file
        test_path = self.base_path / test_file

        print("Loading input files...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        print(f"  -> Train shape: {train_df.shape}")
        print(f"  -> Test shape:  {test_df.shape}")

        return train_df, test_df

    def create_feature_dataframe(
        self,
        original_df: pd.DataFrame,
        feature_matrix: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        feature_df = pd.DataFrame(feature_matrix, columns=feature_names)

        output_df = pd.DataFrame()
        output_df["id"] = original_df["id"]

        if "binary_label" in original_df.columns:
            output_df["binary_label"] = original_df["binary_label"]

        output_df = pd.concat([output_df, feature_df], axis=1)
        return output_df

    def save_data(
        self,
        train_features_df: pd.DataFrame,
        test_features_df: pd.DataFrame,
        train_output_file: str = "train_features.csv",
        test_output_file: str = "test_features.csv"
    ):
        train_output_path = self.base_path / train_output_file
        test_output_path = self.base_path / test_output_file

        train_features_df.to_csv(train_output_path, index=False)
        test_features_df.to_csv(test_output_path, index=False)

        print(f"  -> Saved train features to: {train_output_path}")
        print(f"  -> Saved test features to:  {test_output_path}")

    def run(self):
        train_df, test_df = self.load_data()

        train_texts = train_df["comment_text"].fillna("").astype(str).tolist()
        test_texts = test_df["comment_text"].fillna("").astype(str).tolist()

        print("\nExtracting train features...")
        train_features = self.extractor.fit_transform(train_texts)

        print("Extracting test features...")
        test_features = self.extractor.transform(test_texts)

        feature_names = self.extractor.get_feature_names()

        print(f"  -> Number of features: {len(feature_names)}")

        train_features_df = self.create_feature_dataframe(train_df, train_features, feature_names)
        test_features_df = self.create_feature_dataframe(test_df, test_features, feature_names)

        print("\nSaving output files...")
        self.save_data(train_features_df, test_features_df)

        print("\nFeature extraction completed successfully.")


# =========================
# 5. Interactive Menu
# =========================

def interactive_menu() -> FeatureExtractionPipeline:
    print("=========================================")
    print("   NLP FEATURE EXTRACTION PIPELINE       ")
    print("=========================================")
    print("Choose your feature extraction method:")

    print("\n[1] Feature Extraction Method:")
    print("  1: Bag of Words")
    print("  2: TF-IDF")
    print("  3: Word2Vec")
    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == "1":
        max_features = input("Max features for Bag of Words [default: 5000]: ").strip()
        max_features = int(max_features) if max_features else 5000
        extractor = BagOfWordsExtractor(max_features=max_features)

    elif choice == "2":
        max_features = input("Max features for TF-IDF [default: 5000]: ").strip()
        max_features = int(max_features) if max_features else 5000
        extractor = TfidfExtractor(max_features=max_features)

    elif choice == "3":
        vector_size = input("Word2Vec vector size [default: 100]: ").strip()
        window = input("Word2Vec window size [default: 5]: ").strip()
        min_count = input("Word2Vec min_count [default: 1]: ").strip()
        architecture = input("Architecture: CBOW or Skip-Gram? [c/s, default: c]: ").strip().lower()

        vector_size = int(vector_size) if vector_size else 100
        window = int(window) if window else 5
        min_count = int(min_count) if min_count else 1
        sg = 1 if architecture == "s" else 0

        extractor = Word2VecExtractor(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg
        )

    else:
        print("Invalid choice. Defaulting to TF-IDF with 5000 features.")
        extractor = TfidfExtractor(max_features=5000)

    pipeline = FeatureExtractionPipeline(extractor=extractor, base_path="data")
    return pipeline


# =========================
# 6. Main
# =========================

def main():
    pipeline = interactive_menu()
    pipeline.run()


if __name__ == "__main__":
    main()
