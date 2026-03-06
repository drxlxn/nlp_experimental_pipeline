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


def main():
    mapper = LabelMapper()
    mapper.run()


if __name__ == "__main__":
    main()