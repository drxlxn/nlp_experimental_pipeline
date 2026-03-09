import os
import json
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split

from LabelMapping import LabelMapper
from DataPreprocessing import (
    DataPreprocessor, NLTKTokenizer, SpacyTokenizer,
    NLTKStopWordRemover, SpacyStopWordRemover,
    NLTKPorterStemmer, NLTKSnowballStemmer
)
from FeatureExtraction import (
    BagOfWordsExtractor, TfidfExtractor, Word2VecExtractor
)
from ModelTraining import (
    LogisticRegressionModel, SVMModel, PyTorchDNNModel, ModelTrainer
)


class ExperimentRunner:
    def __init__(self):
        self.config = {
            "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "preprocessing": {},
            "feature_extraction": {},
            "model": {}
        }
        self.data_dir = "data"
        self.models_dir = "models"

        self.preprocessor = None
        self.extractor = None

        # Model Tracking
        self.model_strategy = None
        self.do_tune = False
        self.do_plot = False

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

    def ui_build_model(self):
        print("\n--- [STEP 4] Model Selection ---")
        print("1: Logistic Regression")
        print("2: Support Vector Machine (Linear)")
        print("3: Deep Neural Network (PyTorch)")
        choice = input("Choice [1]: ").strip()

        if choice == "3":
            dev = input("Device (auto, cpu, gpu) [auto]: ").strip() or "auto"
            self.model_strategy = PyTorchDNNModel(device_preference=dev)
            self.config["model"] = {"type": "PyTorchDNN", "device": dev}
        elif choice == "2":
            self.model_strategy = SVMModel()
            self.config["model"] = {"type": "SVM"}
        else:
            self.model_strategy = LogisticRegressionModel()
            self.config["model"] = {"type": "LogisticRegression"}

        tune = input("Tune hyperparameters? (y/n) [n]: ").strip().lower()
        self.do_tune = (tune == 'y')
        self.config["model"]["tuned"] = self.do_tune

        plot = input("Generate learning curve plot? (y/n) [n]: ").strip().lower()
        self.do_plot = (plot == 'y')

    def execute_pipeline(self):
        print("\n--- [STEP 5] Data Pipeline Execution ---")
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

        if not train_texts:
            raise ValueError("All training data was filtered out during preprocessing.")

        print("Processing text...")
        train_tokens = self.preprocessor.process_dataset(train_texts)
        test_tokens = self.preprocessor.process_dataset(test_texts)

        print("Extracting features...")
        train_feats = self.extractor.fit_transform(train_tokens)
        test_feats = self.extractor.transform(test_tokens)

        print("\n--- [STEP 6] Saving Features ---")
        names = [f"feat_{name}" for name in self.extractor.get_feature_names()]
        exp_id = self.config['experiment_id']

        # 1. Build the Training DataFrame
        train_final = pd.concat([
            pd.DataFrame({"id": train_ids, "binary_label": train_labels}),
            pd.DataFrame(train_feats, columns=names)
        ], axis=1)

        # 2. Build the Testing DataFrame
        test_out_dict = {"id": test_ids}
        if has_test_labels:
            test_out_dict["binary_label"] = test_labels

        test_final = pd.concat([
            pd.DataFrame(test_out_dict),
            pd.DataFrame(test_feats, columns=names)
        ], axis=1)

        # 3. Save both as Parquet files
        train_out_path = os.path.join(self.data_dir, f"train_features_{exp_id}.parquet")
        test_out_path = os.path.join(self.data_dir, f"test_features_{exp_id}.parquet")

        train_final.to_parquet(train_out_path, engine='pyarrow')
        test_final.to_parquet(test_out_path, engine='pyarrow')

        print(f"Features saved with ID: {exp_id}")

        print("\n--- [STEP 7] Model Training & Evaluation ---")
        # Split 80/20 for local evaluation to ensure we get an accurate F1 score
        print("Splitting training data for validation (80/20 split)...")
        X_train, X_val, y_train, y_val = train_test_split(
            train_feats, train_labels, test_size=0.2, random_state=42
        )

        trainer = ModelTrainer(self.model_strategy)
        plot_dir = self.models_dir if self.do_plot else None

        # Determine correct file extension for saving
        ext = ".pth" if isinstance(self.model_strategy, PyTorchDNNModel) else ".pkl"
        model_save_path = os.path.join(self.models_dir, f"model_{exp_id}{ext}")

        metrics, best_params = trainer.run_training(
            X_train=X_train, y_train=y_train,
            X_test=X_val, y_test=y_val,
            do_tune=self.do_tune,
            model_save_path=model_save_path,
            plot_dir=plot_dir,
            experiment_name=f"Exp_{exp_id}"
        )

        # Log results to config
        self.config["results"] = metrics
        if best_params:
            self.config["model"]["best_params"] = best_params

        self.save_experiment()
        print("\nPipeline execution fully complete!")

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
        self.ui_build_model()
        self.execute_pipeline()


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()