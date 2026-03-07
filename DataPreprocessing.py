from typing import List
import nltk
from nltk.tokenize import word_tokenize
import spacy
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer, SnowballStemmer
import numpy as np
#
# 1. The Blueprints (Strategy Interface)
#
class TokenizerStrategy:
    def tokenize(self, texts: List[str]) -> List[List[str]]:
        raise NotImplementedError()

class StopWordStrategy:
    def remove_stopwords(self, tokenized_dataset: List[List[str]]) -> List[List[str]]:
        raise NotImplementedError()

class StemmerStrategy:
    def stem(self, tokenized_dataset: List[List[str]]) -> List[List[str]]:
        raise NotImplementedError("Subclasses must implement the stem method")


# ============
# Tokenizers
# ============

class NLTKTokenizer(TokenizerStrategy):
    def tokenize(self, texts: List[str]) -> List[List[str]]:

        return [word_tokenize(text) for text in texts]


class SpacyTokenizer(TokenizerStrategy):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenizes a whole list of strings extremely fast."""
        tokenized_dataset = []

        # nlp.pipe() is spaCy's secret weapon for processing lists of text.
        # Using 'disable' to turn off the grammar and entity tools and only focusing on the tokenization
        for doc in self.nlp.pipe(texts, disable=["tagger", "parser", "ner", "lemmatizer"]):
            # doc is a spaCy object, we just want the raw string text of each token
            tokens = [token.text for token in doc]
            tokenized_dataset.append(tokens)

        return tokenized_dataset

# =================
# StopwordRemovers
# =================

class NLTKStopWordRemover(StopWordStrategy):
    def __init__(self, language: str = 'english'):
        # Download the stopwords list if not already present
        nltk.download('stopwords', quiet=True)
        # Convert to a set for faster lookups
        self.stop_words = set(stopwords.words(language))

    def remove_stopwords(self, tokenized_dataset: List[List[str]]) -> List[List[str]]:
        filtered_dataset = []

        for tokens in tokenized_dataset:
            # Only keep the word if it's NOT in our stop words set
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
            filtered_dataset.append(filtered_tokens)

        return filtered_dataset


class SpacyStopWordRemover(StopWordStrategy):
    def __init__(self):
        self.stop_words = set(STOP_WORDS)

    def remove_stopwords(self, tokenized_dataset: List[List[str]]) -> List[List[str]]:
        filtered_dataset = []

        for tokens in tokenized_dataset:
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
            filtered_dataset.append(filtered_tokens)

        return filtered_dataset


# ==========
# Stemmers
# ==========

class NLTKPorterStemmer(StemmerStrategy):
    def __init__(self):
        # Oldest and most famous
        self.stemmer = PorterStemmer()

    def stem(self, tokenized_dataset: List[List[str]]) -> List[List[str]]:
        stemmed_dataset = []

        for tokens in tokenized_dataset:
            # Apply the stemmer to every single word in the token list
            stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
            stemmed_dataset.append(stemmed_tokens)

        return stemmed_dataset


class NLTKSnowballStemmer(StemmerStrategy):
    def __init__(self, language: str = 'english'):
        # More aggressive and accurate stemmer
        self.stemmer = SnowballStemmer(language)

    def stem(self, tokenized_dataset: List[List[str]]) -> List[List[str]]:
        stemmed_dataset = []

        for tokens in tokenized_dataset:
            stemmed_tokens = [self.stemmer.stem(word) for word in tokens]
            stemmed_dataset.append(stemmed_tokens)

        return stemmed_dataset





# 3. The Main Preprocessing Stage

class DataPreprocessor:
    """Handles the full preprocessing pipeline for text data."""

    def __init__(self, tokenizer: TokenizerStrategy, stopwordremover: StopWordStrategy = None, stemmer: StemmerStrategy = None):
        # We "inject" the chosen Lego block here
        self.tokenizer = tokenizer
        self.stop_word_remover = stopwordremover
        self.stemmer = stemmer

    def remove_duplicates(self, texts: List[str], labels: List[int], ids: List[str]):
        """Removes duplicate texts while keeping the labels and IDs perfectly aligned."""
        print("Removing duplicates...")
        seen_texts = set()
        unique_texts = []
        unique_labels = []
        unique_ids = []

        for text, label, item_id in zip(texts, labels, ids):
            text_lower = text.lower()

            if text_lower not in seen_texts:
                seen_texts.add(text_lower)
                unique_texts.append(text)
                unique_labels.append(label)
                unique_ids.append(item_id)

        duplicates_removed = len(texts) - len(unique_texts)
        print(f"  -> Removed {duplicates_removed} duplicate entries.")

        return unique_texts, unique_labels, unique_ids

    def detect_outliers(self, texts: List[str], labels: List[int], ids: List[str], lower_percentile: float = 1.0,
                        upper_percentile: float = 95.0):
        """Removes outlier texts based on word count percentiles while keeping labels and IDs aligned."""
        print(
            f"Removing outliers (keeping texts between the {lower_percentile}th and {upper_percentile}th percentiles)...")

        word_counts = [len(text.split()) for text in texts]
        min_words = np.percentile(word_counts, lower_percentile)
        max_words = np.percentile(word_counts, upper_percentile)

        print(f"  -> Calculated thresholds: Min {int(min_words)} words, Max {int(max_words)} words.")

        filtered_texts = []
        filtered_labels = []
        filtered_ids = []

        for text, label, item_id, count in zip(texts, labels, ids, word_counts):
            if min_words <= count <= max_words:
                filtered_texts.append(text)
                filtered_labels.append(label)
                filtered_ids.append(item_id)

        outliers_removed = len(texts) - len(filtered_texts)
        print(f"  -> Removed {outliers_removed} outlier entries.")

        return filtered_texts, filtered_labels, filtered_ids

    def process_dataset(self, texts: List[str]) -> List[List[str]]:
        # 1. Tokenize
        tokens = self.tokenizer.tokenize(texts)

        # 2. Remove Stop Words
        if self.stop_word_remover:
            tokens = self.stop_word_remover.remove_stopwords(tokens)

        # 3. Stem
        if self.stemmer:
            tokens = self.stemmer.stem(tokens)

        return tokens


# ==========================================
# 4. How the User (or your Main Runner) uses it
# ==========================================
# Assuming this is at the bottom of DataPreprocessing.py
# (or in a new file where you import these classes)

def interactive_menu():
    print("=========================================")
    print("   NLP EXPERIMENTAL PIPELINE BUILDER     ")
    print("=========================================")
    print("Let's build your preprocessing pipeline!")

    # 1. Choose Tokenizer
    print("\n[1] Choose Tokenizer (Required):")
    print("  1: NLTK Tokenizer")
    print("  2: spaCy Tokenizer")
    tok_choice = input("Enter choice (1 or 2): ")

    if tok_choice == '2':
        tokenizer = SpacyTokenizer()
    else:
        tokenizer = NLTKTokenizer()  # Default to NLTK

    # 2. Choose Stop Word Remover
    print("\n[2] Choose Stop Word Remover (Optional):")
    print("  1: NLTK Stop Words")
    print("  2: spaCy Stop Words")
    print("  3: Skip Stop Word Removal")
    sw_choice = input("Enter choice (1, 2, or 3): ")

    if sw_choice == '1':
        stopword_remover = NLTKStopWordRemover()
    elif sw_choice == '2':
        stopword_remover = SpacyStopWordRemover()
    else:
        stopword_remover = None

    # 3. Choose Stemmer
    print("\n[3] Choose Stemmer (Optional):")
    print("  1: NLTK Porter Stemmer")
    print("  2: NLTK Snowball Stemmer")
    print("  3: Skip Stemming")
    stem_choice = input("Enter choice (1, 2, or 3): ")

    if stem_choice == '1':
        stemmer = NLTKPorterStemmer()
    elif stem_choice == '2':
        stemmer = NLTKSnowballStemmer()
    else:
        stemmer = None

    # 4. Build the Pipeline!
    print("\n=========================================")
    print("Assembling your custom pipeline...")

    preprocessor = DataPreprocessor(
        tokenizer=tokenizer,
        stopwordremover=stopword_remover,
        stemmer=stemmer
    )

    return preprocessor






if __name__ == "__main__":
    # 1. Dummy raw data
    raw_texts = [
        "This is a terrible hate speech sample comment.",
        "Another completely normal Wikipedia comment.",
        "This is a terrible hate speech sample comment.",  # Duplicate!
        "k",  # Outlier (too short)
        "The runners are running really fast today!"
    ]
    raw_labels = [1, 0, 1, 0, 0]

    # 2. Let the user build their pipeline
    user_preprocessor = interactive_menu()

    # 3. Run the dataset-level cleaning first
    print("\nRunning dataset-level cleaning...")
    clean_texts, clean_labels = user_preprocessor.remove_duplicates(raw_texts, raw_labels)
    clean_texts, clean_labels = user_preprocessor.detect_outliers(clean_texts, clean_labels)

    # 4. Run the text-level string processing
    print("\nProcessing text data...")
    final_output = user_preprocessor.process_dataset(clean_texts)

    print("\n=== FINAL PROCESSED OUTPUT ===")
    for original, processed in zip(clean_texts, final_output):
        print(f"Original:  {original}")
        print(f"Processed: {processed}\n")