"""
=========================================================
Research Grade Exploratory Data Analysis (EDA)
Trustworthy Anxiety Framework

Author : Rehana N
=========================================================

This script performs complete exploratory data analysis
for text classification datasets.

Outputs:
--------
results/
    figures/
    tables/

Compatible with:
- MentalDistress Dataset
- Multiclass Social Media Dataset
- Any text classification dataset
=========================================================
"""

import os
import re
import string
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

FIGURE_DIR = "results/figures"
TABLE_DIR = "results/tables"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)


# =========================================================
# MAIN CLASS
# =========================================================

class TextDatasetEDA:

    def __init__(
        self,
        csv_path,
        text_column,
        label_column,
        dataset_name
    ):

        self.csv_path = csv_path
        self.text_column = text_column
        self.label_column = label_column
        self.dataset_name = dataset_name

        self.df = None

    # =====================================================
    # LOAD DATASET
    # =====================================================

    def load_dataset(self):

        print("=" * 70)
        print(f"Loading {self.dataset_name}")
        print("=" * 70)

        self.df = pd.read_csv(self.csv_path)

        print(f"Rows    : {len(self.df)}")
        print(f"Columns : {len(self.df.columns)}")

    # =====================================================
    # BASIC CLEANING
    # =====================================================

    def preprocess(self):

        print("\nCleaning Dataset...")

        self.df = self.df.dropna(
            subset=[
                self.text_column,
                self.label_column
            ]
        ).copy()

        self.df[self.text_column] = (
            self.df[self.text_column]
            .astype(str)
            .str.strip()
        )

        self.df = self.df[
            self.df[self.text_column] != ""
        ]

        print("Cleaning Completed.")

    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    def dataset_summary(self):

        print("\nDataset Summary")

        print("-" * 40)

        print("Shape :", self.df.shape)

        print("\nColumns")

        print(self.df.columns.tolist())

        print("\nData Types")

        print(self.df.dtypes)

        print("\nMissing Values")

        print(self.df.isnull().sum())

        duplicate_rows = self.df.duplicated().sum()

        duplicate_text = (
            self.df[self.text_column]
            .duplicated()
            .sum()
        )

        print("\nDuplicate Rows :", duplicate_rows)

        print("Duplicate Text :", duplicate_text)

        summary = pd.DataFrame({

            "Metric": [

                "Rows",
                "Columns",
                "Missing Rows",
                "Duplicate Rows",
                "Duplicate Text",
                "Memory Usage (MB)"

            ],

            "Value": [

                len(self.df),

                len(self.df.columns),

                self.df.isnull().any(axis=1).sum(),

                duplicate_rows,

                duplicate_text,

                round(
                    self.df.memory_usage(
                        deep=True
                    ).sum() / 1024 / 1024,
                    2
                )

            ]

        })

        summary.to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_dataset_summary.csv"

            ),

            index=False

        )

    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    def class_distribution(self):

        print("\nClass Distribution")

        counts = (
            self.df[self.label_column]
            .value_counts()
        )

        percentages = (

            self.df[self.label_column]

            .value_counts(normalize=True)

            * 100

        ).round(2)

        distribution = pd.DataFrame({

            "Count": counts,

            "Percentage": percentages

        })

        print(distribution)

        distribution.to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_class_distribution.csv"

            )

        )

        plt.figure(figsize=(8,5))

        counts.plot(

            kind="bar"

        )

        plt.title(

            f"{self.dataset_name} Class Distribution"

        )

        plt.xlabel("Class")

        plt.ylabel("Samples")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                FIGURE_DIR,

                f"{self.dataset_name}_class_distribution.png"

            )

        )

        plt.close()
            # =====================================================
    # TEXT STATISTICS
    # =====================================================

    def text_statistics(self):

        print("\nCalculating Text Statistics...")

        self.df["character_count"] = (
            self.df[self.text_column]
            .astype(str)
            .apply(len)
        )

        self.df["word_count"] = (
            self.df[self.text_column]
            .astype(str)
            .apply(lambda x: len(x.split()))
        )

        self.df["sentence_count"] = (
            self.df[self.text_column]
            .astype(str)
            .apply(
                lambda x: len(
                    re.findall(r"[.!?]+", x)
                ) or 1
            )
        )

        statistics = pd.DataFrame({

            "Metric": [

                "Average Characters",
                "Median Characters",
                "Maximum Characters",
                "Minimum Characters",

                "Average Words",
                "Median Words",
                "Maximum Words",
                "Minimum Words",

                "Average Sentences",
                "Maximum Sentences"

            ],

            "Value": [

                round(
                    self.df["character_count"].mean(),2
                ),

                self.df["character_count"].median(),

                self.df["character_count"].max(),

                self.df["character_count"].min(),

                round(
                    self.df["word_count"].mean(),2
                ),

                self.df["word_count"].median(),

                self.df["word_count"].max(),

                self.df["word_count"].min(),

                round(
                    self.df["sentence_count"].mean(),2
                ),

                self.df["sentence_count"].max()

            ]

        })

        print(statistics)

        statistics.to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_text_statistics.csv"

            ),

            index=False

        )

    # =====================================================
    # TEXT DISTRIBUTION PLOTS
    # =====================================================

    def text_length_plots(self):

        print("\nGenerating Text Distribution Graphs...")

        columns = [

            "character_count",

            "word_count",

            "sentence_count"

        ]

        for column in columns:

            plt.figure(figsize=(8,5))

            plt.hist(

                self.df[column],

                bins=40,

                edgecolor="black"

            )

            plt.title(

                f"{self.dataset_name} - {column}"

            )

            plt.xlabel(column)

            plt.ylabel("Frequency")

            plt.tight_layout()

            plt.savefig(

                os.path.join(

                    FIGURE_DIR,

                    f"{self.dataset_name}_{column}.png"

                )

            )

            plt.close()

        plt.figure(figsize=(5,6))

        plt.boxplot(

            self.df["word_count"],

            vert=True

        )

        plt.title(

            f"{self.dataset_name} Word Count"

        )

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                FIGURE_DIR,

                f"{self.dataset_name}_word_boxplot.png"

            )

        )

        plt.close()

    # =====================================================
    # VOCABULARY ANALYSIS
    # =====================================================

    def vocabulary_analysis(self):

        print("\nVocabulary Analysis...")

        text = " ".join(

            self.df[self.text_column]

            .astype(str)

            .tolist()

        )

        words = re.findall(

            r"\b[a-zA-Z']+\b",

            text.lower()

        )

        vocabulary = Counter(words)

        vocabulary_size = len(vocabulary)

        print(

            f"Vocabulary Size : {vocabulary_size}"

        )

        top_words = pd.DataFrame(

            vocabulary.most_common(30),

            columns=[

                "Word",

                "Frequency"

            ]

        )

        top_words.to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_top_words.csv"

            ),

            index=False

        )

    # =====================================================
    # BIGRAM ANALYSIS
    # =====================================================

    def bigram_analysis(self):

        print("\nBigram Analysis...")

        vectorizer = CountVectorizer(

            stop_words="english",

            ngram_range=(2,2),

            max_features=30

        )

        matrix = vectorizer.fit_transform(

            self.df[self.text_column]

        )

        bigrams = pd.DataFrame({

            "Bigram":

                vectorizer.get_feature_names_out(),

            "Frequency":

                matrix.sum(axis=0).A1

        })

        bigrams = bigrams.sort_values(

            by="Frequency",

            ascending=False

        )

        bigrams.to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_top_bigrams.csv"

            ),

            index=False

        )

        print(bigrams.head(10))

    # =====================================================
    # WORD CLOUD
    # =====================================================

    def wordcloud(self):

        print("\nGenerating Word Cloud...")

        text = " ".join(

            self.df[self.text_column]

            .astype(str)

            .tolist()

        )

        cloud = WordCloud(

            width=1400,

            height=700,

            background_color="white",

            max_words=300

        ).generate(text)

        plt.figure(figsize=(14,7))

        plt.imshow(cloud)

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                FIGURE_DIR,

                f"{self.dataset_name}_wordcloud.png"

            )

        )

        plt.close()
            # =====================================================
    # DATASET QUALITY REPORT
    # =====================================================

    def dataset_quality(self):

        print("\nDataset Quality Report...")

        empty_text = (
            self.df[self.text_column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        duplicate_text = (
            self.df[self.text_column]
            .duplicated()
            .sum()
        )

        short_text = (
            self.df["word_count"] < 5
        ).sum()

        long_text = (
            self.df["word_count"] > 200
        ).sum()

        report = pd.DataFrame({

            "Metric":[

                "Empty Text",

                "Duplicate Text",

                "Short Text (<5 words)",

                "Long Text (>200 words)"

            ],

            "Value":[

                empty_text,

                duplicate_text,

                short_text,

                long_text

            ]

        })

        print(report)

        report.to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_quality_report.csv"

            ),

            index=False

        )

    # =====================================================
    # RANDOM SAMPLE INSPECTION
    # =====================================================

    def sample_examples(self):

        print("\nSaving Sample Examples...")

        samples = []

        labels = self.df[self.label_column].unique()

        for label in labels:

            subset = self.df[

                self.df[self.label_column] == label

            ]

            n = min(3, len(subset))

            samples.append(

                subset.sample(

                    n=n,

                    random_state=42

                )[[

                    self.text_column,

                    self.label_column

                ]]

            )

        pd.concat(samples).to_csv(

            os.path.join(

                TABLE_DIR,

                f"{self.dataset_name}_sample_examples.csv"

            ),

            index=False

        )

    # =====================================================
    # COMPLETE PIPELINE
    # =====================================================

    def run(self):

        self.load_dataset()

        self.preprocess()

        self.dataset_summary()

        self.class_distribution()

        self.text_statistics()

        self.text_length_plots()

        self.vocabulary_analysis()

        self.bigram_analysis()

        self.wordcloud()

        self.dataset_quality()

        self.sample_examples()

        print("\n" + "="*70)

        print(f"{self.dataset_name} Analysis Completed Successfully")

        print("="*70)


# =========================================================
# MAIN
# =========================================================

def main():

    datasets = [

        {

            "csv_path":

            "data/raw/MentalDistress/Mental Distress Dataset-original.csv",

            "text_column":"text",

            "label_column":"label",

            "dataset_name":"MentalDistress"

        },

        {

            "csv_path":

            "data/raw/MulticlassSocialMedia/Mental_Health_6Class_Final_Cleaned.csv",

            "text_column":"Text",

            "label_column":"Emotion",

            "dataset_name":"MulticlassSocialMedia"

        }

    ]

    for dataset in datasets:

        analyzer = TextDatasetEDA(

            csv_path=dataset["csv_path"],

            text_column=dataset["text_column"],

            label_column=dataset["label_column"],

            dataset_name=dataset["dataset_name"]

        )

        analyzer.run()

        print("\n")


if __name__ == "__main__":

    main()