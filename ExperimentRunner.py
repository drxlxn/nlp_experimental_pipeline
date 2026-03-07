import os
import json
import pandas as pd
from datetime import datetime

from LabelMapping import LabelMapper
from DataPreprocessing import (
    DataPreprocessor, NLTKTokenizer, SpacyTokenizer,
    NLTKStopWordRemover, SpacyStopWordRemover,
    NLTKPorterStemmer, NLTKSnowballStemmer
)
from FeatureExtraction import (
    BagOfWordsExtractor, TfidfExtractor, Word2VecExtractor
)


class ExperimentRunner:
    def __init__(self):
        self.config = {
            "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "preprocessing": {},
            "feature_extraction": {}
        }
        self.data_dir = "data"
        self.models_dir = "models"

        self.preprocessor = None
        self.extractor = None

        self.train_path = None
        self.test_path = None

    def setup_directories(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def ui_data_preparation(self):
        print("\n--- [STEP 1] Data Preparation ---")
        self.train_path = os.path.join(self.data_dir, "train_binary_labels.csv")
        self.test_path = os.path.join(self.data_dir, "test_binary_labels.csv")

        if os.path.exists(self.train_path) and os.path.exists(self.test_path):
            print(f"Clean datasets found in '{self.data_dir}'.")
            if input("Re-run label mapping? (y/n) [n]: ").strip().lower() != 'y':
                return

        path = input("Enter path to raw Kaggle CSVs: ").strip()
        if path and not path.endswith("/"):
            path += "/"

        mapper = LabelMapper()
        mapper.run(
            train_input_path=f"{path}train.csv",
            train_output_path=self.train_path,
            test_input_path=f"{path}test.csv",
            test_labels_input_path=f"{path}test_labels.csv",
            test_output_path=self.test_path
        )

    def ui_build_preprocessor(self):
        print("\n--- [STEP 2] Preprocessing Configuration ---")

        t_choice = input("Tokenizer (1: NLTK, 2: spaCy) [1]: ").strip()
        self.config["preprocessing"]["tokenizer"] = "spaCy" if t_choice == '2' else "NLTK"
        tokenizer = SpacyTokenizer() if t_choice == '2' else NLTKTokenizer()

        s_choice = input("Stop Words (1: NLTK, 2: spaCy, 3: None) [3]: ").strip()
        sw = None
        if s_choice == '1':
            sw = NLTKStopWordRemover()
            self.config["preprocessing"]["stop_words"] = "NLTK"
        elif s_choice == '2':
            sw = SpacyStopWordRemover()
            self.config["preprocessing"]["stop_words"] = "spaCy"
        else:
            self.config["preprocessing"]["stop_words"] = "None"

        m_choice = input("Stemmer (1: Porter, 2: Snowball, 3: None) [3]: ").strip()
        stem = None
        if m_choice == '1':
            stem = NLTKPorterStemmer()
            self.config["preprocessing"]["stemmer"] = "Porter"
        elif m_choice == '2':
            stem = NLTKSnowballStemmer()
            self.config["preprocessing"]["stemmer"] = "Snowball"
        else:
            self.config["preprocessing"]["stemmer"] = "None"

        self.preprocessor = DataPreprocessor(tokenizer, sw, stem)

    def ui_build_extractor(self):
        print("\n--- [STEP 3] Feature Extraction Configuration ---")
        print("1: Bag of Words\n2: TF-IDF\n3: Word2Vec")
        choice = input("Choice [2]: ").strip()

        if choice == "1":
            feat = input("Max features [5000]: ").strip()
            feat_val = int(feat) if feat else 5000
            self.config["feature_extraction"] = {"method": "BagOfWords", "max_features": feat_val}
            self.extractor = BagOfWordsExtractor(max_features=feat_val)

        elif choice == "3":
            size = input("Vector size [100]: ").strip()
            size_val = int(size) if size else 100
            self.config["feature_extraction"] = {"method": "Word2Vec", "vector_size": size_val}
            self.extractor = Word2VecExtractor(vector_size=size_val)

        else:
            feat = input("Max features [5000]: ").strip()
            feat_val = int(feat) if feat else 5000
            self.config["feature_extraction"] = {"method": "TF-IDF", "max_features": feat_val}
            self.extractor = TfidfExtractor(max_features=feat_val)

    def execute_pipeline(self):
        print("\n--- [STEP 4] Execution ---")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_texts = train_df["comment_text"].fillna("").tolist()
        train_labels = train_df["binary_label"].tolist()
        train_ids = train_df["id"].tolist()

        test_texts = test_df["comment_text"].fillna("").tolist()
        test_ids = test_df["id"].tolist()

        has_test_labels = "binary_label" in test_df.columns
        if has_test_labels:
            test_labels = test_df["binary_label"].tolist()

        print("Cleaning training data...")
        train_texts, train_labels, train_ids = self.preprocessor.remove_duplicates(
            train_texts, train_labels, train_ids
        )

        train_texts, train_labels, train_ids = self.preprocessor.detect_outliers(
            train_texts, train_labels, train_ids, lower_percentile=1.0, upper_percentile=99.0
        )

        # Guard against empty datasets after filtering
        if not train_texts:
            raise ValueError("All training data was filtered out during preprocessing. Check your outlier percentiles.")

        print("Processing text...")
        train_tokens = self.preprocessor.process_dataset(train_texts)
        test_tokens = self.preprocessor.process_dataset(test_texts)

        print("Extracting features...")
        train_feats = self.extractor.fit_transform(train_tokens)
        test_feats = self.extractor.transform(test_tokens)

        print("\n--- [STEP 5] Saving Features & Config ---")
        names = self.extractor.get_feature_names()

        # Reconstruct without the 'comment_text' string column
        train_final = pd.concat([
            pd.DataFrame({"id": train_ids, "binary_label": train_labels}),
            pd.DataFrame(train_feats, columns=names)
        ], axis=1)

        test_out_dict = {"id": test_ids}
        if has_test_labels:
            test_out_dict["binary_label"] = test_labels

        test_final = pd.concat([
            pd.DataFrame(test_out_dict),
            pd.DataFrame(test_feats, columns=names)
        ], axis=1)

        exp_id = self.config['experiment_id']

        train_out_path = os.path.join(self.data_dir, f"train_features_{exp_id}.csv")
        test_out_path = os.path.join(self.data_dir, f"test_features_{exp_id}.csv")

        train_final.to_csv(train_out_path, index=False)
        test_final.to_csv(test_out_path, index=False)

        self.save_experiment()
        print(f"Process complete. Features saved with ID: {exp_id}")

    def save_experiment(self):
        config_path = os.path.join(self.models_dir, f"config_{self.config['experiment_id']}.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"Experiment configuration saved to: {config_path}")

    def run(self):
        print("Initializing Pipeline")
        self.setup_directories()
        self.ui_data_preparation()
        self.ui_build_preprocessor()
        self.ui_build_extractor()
        self.execute_pipeline()


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()