import sys
import argparse

sys.path.insert(0, "src")
from pipeline import pipeline


def main():
    parser = argparse.ArgumentParser(description="Synthetic OCR Image Generator")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Hub repository ID for dataset upload",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        required=True,
        help="Font file path for character similarity DB generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Base directory for all generated data",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        help="Language code (e.g., ko, en)",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=10000,
        help="Number of sentences to extract from Wikipedia",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--typo-ratio",
        type=float,
        default=0.15,
        help="Ratio of words to introduce typos",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="sentence",
        choices=["sentence", "table", "document", "markdown", "kie"],
        help="Format of images to generate (sentence, table, document, markdown, or kie)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template for generation. Table: invoice, receipt, form, letter, report. KIE: receipt, invoice, form, business_card",
    )
    parser.add_argument(
        "--table-size",
        type=str,
        default="3-8",
        help="Table size range as 'min_rows-max_cols' (e.g., '3-8' for 3-8 rows and cols)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        default=False,
        help="Generate mixed format dataset (sentence, table, document)",
    )

    args = parser.parse_args()
    pipeline(**vars(args))


if __name__ == "__main__":
    main()
