import pandas as pd
import numpy as np


class LabelMapper:
    LABEL_COLUMNS = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    def __init__(self):
        pass

    def map_labels(self, row):
        values = [int(row[col]) for col in self.LABEL_COLUMNS]

        if any(value == -1 for value in values):
            return np.nan
        if any(value == 1 for value in values):
            return 1
        return 0

    def create_train_binary_labels(
        self,
        input_path="train.csv",
        output_path="train_binary_labels.csv",
    ):
        train_df = pd.read_csv(input_path)

        train_binary_df = train_df[["id", "comment_text"]].copy()
        train_binary_df["binary_label"] = train_df.apply(self.map_labels, axis=1)

        train_binary_df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(train_binary_df)} rows.")

    def create_test_binary_labels(
        self,
        test_input_path="test.csv",
        test_labels_input_path="test_labels.csv",
        output_path="test_binary_labels.csv",
    ):
        test_df = pd.read_csv(test_input_path)
        test_labels_df = pd.read_csv(test_labels_input_path)

        test_labels_binary_df = test_labels_df[["id"]].copy()
        test_labels_binary_df["binary_label"] = test_labels_df.apply(self.map_labels, axis=1)

        test_binary_df = test_df[["id", "comment_text"]].merge(
            test_labels_binary_df,
            on="id",
            how="left",
        )

        if test_binary_df["binary_label"].isna().all():
            print("Warning: All target values are NaN.")
        elif test_binary_df["binary_label"].isna().any():
            missing_count = test_binary_df["binary_label"].isna().sum()
            print(f"Warning: {missing_count} rows contain NaN as target variable.")

        test_binary_df = test_binary_df.dropna(subset=['binary_label'])
        test_binary_df["binary_label"] = test_binary_df["binary_label"].astype(int)
        test_binary_df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(test_binary_df)} rows.")

    def run(
        self,
        train_input_path="data/train.csv",
        train_output_path="data/train_binary_labels.csv",
        test_input_path="data/test.csv",
        test_labels_input_path="data/test_labels.csv",
        test_output_path="data/test_binary_labels.csv",
    ):
        self.create_train_binary_labels(
            input_path=train_input_path,
            output_path=train_output_path,
        )
        self.create_test_binary_labels(
            test_input_path=test_input_path,
            test_labels_input_path=test_labels_input_path,
            output_path=test_output_path,
        )


import os


def main():
    print("=========================================")
    print("      LABEL MAPPING & SETUP UTILITY      ")
    print("=========================================")

    # 1. Where are the raw files coming from?
    input_dir = input("1. Enter path to raw Kaggle CSVs (e.g., /.../Project_00/data/): ").strip()
    if not input_dir.endswith("/") and input_dir != "":
        input_dir += "/"

    # 2. Where do you want to build the project structure?
    output_base_dir = input(
        "2. Enter path to create your project folders (or press Enter for current directory): ").strip()
    if output_base_dir == "":
        output_base_dir = "./"  # Defaults to current folder
    elif not output_base_dir.endswith("/"):
        output_base_dir += "/"

    # 3. Create the smart subfolders!
    data_out_dir = os.path.join(output_base_dir, "data")
    models_out_dir = os.path.join(output_base_dir, "models")

    # exist_ok=True prevents crashes if the folders already exist
    os.makedirs(data_out_dir, exist_ok=True)
    os.makedirs(models_out_dir, exist_ok=True)

    print(f"\n📁 Verified project structure:")
    print(f"   ├─ {data_out_dir}/")
    print(f"   └─ {models_out_dir}/")
    print("\nRunning label mapping... (This takes about 30 seconds ☕)")

    # 4. Run the mapper and point the outputs to the new data folder!
    mapper = LabelMapper()
    mapper.run(
        train_input_path=f"{input_dir}train.csv",
        train_output_path=f"{data_out_dir}/train_binary_labels.csv",
        test_input_path=f"{input_dir}test.csv",
        test_labels_input_path=f"{input_dir}test_labels.csv",
        test_output_path=f"{data_out_dir}/test_binary_labels.csv"
    )

    print(f"\n✅ Success! Clean data saved in '{data_out_dir}'")


if __name__ == "__main__":
    main()
