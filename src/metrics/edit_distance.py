"""Edit distance based metrics for OCR evaluation."""

from typing import Callable, List, Optional, Union


def levenshtein_distance(s1: Union[str, List], s2: Union[str, List]) -> int:
    """
    Calculates the Levenshtein distance between two sequences.
    """
    if not s2 or len(s2) == 0:
        return len(s1)

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def cer(reference: str, hypothesis: str) -> float:
    """
    Calculates the Character Error Rate (CER).
    CER = (Substitutions + Insertions + Deletions) / Total number of characters in reference
    """
    # Handle None values
    if hypothesis is None:
        hypothesis = ""
    if reference is None:
        reference = ""

    if not reference:
        return 1.0 if hypothesis else 0.0
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """
    Calculates the Word Error Rate (WER).
    WER = (Substitutions + Insertions + Deletions) / Total number of words in reference
    """
    # Handle None values
    if hypothesis is None:
        hypothesis = ""
    if reference is None:
        reference = ""

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def normalized_cer(
    reference: str,
    hypothesis: str,
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> float:
    """
    Calculate CER with optional text normalization.

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.
        normalize_fn: Optional normalization function.

    Returns:
        Normalized CER value.
    """
    if normalize_fn:
        reference = normalize_fn(reference)
        hypothesis = normalize_fn(hypothesis)
    return cer(reference, hypothesis)


def normalized_wer(
    reference: str,
    hypothesis: str,
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> float:
    """
    Calculate WER with optional text normalization.

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.
        normalize_fn: Optional normalization function.

    Returns:
        Normalized WER value.
    """
    if normalize_fn:
        reference = normalize_fn(reference)
        hypothesis = normalize_fn(hypothesis)
    return wer(reference, hypothesis)


def accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate exact match accuracy.

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    # Handle None values
    if hypothesis is None:
        hypothesis = ""
    if reference is None:
        reference = ""
    return 1.0 if reference.strip() == hypothesis.strip() else 0.0


def word_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate word-level accuracy (percentage of matching words).

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.

    Returns:
        Proportion of reference words found in hypothesis.
    """
    # Handle None values
    if hypothesis is None:
        hypothesis = ""
    if reference is None:
        reference = ""

    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())

    if not ref_words:
        return 1.0 if not hyp_words else 0.0

    return len(ref_words & hyp_words) / len(ref_words)


def character_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate character-level accuracy (1 - CER, clamped to [0, 1]).

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.

    Returns:
        Character accuracy value.
    """
    # Handle None values (cer already handles this, but be explicit)
    if hypothesis is None:
        hypothesis = ""
    if reference is None:
        reference = ""
    return max(0.0, 1.0 - cer(reference, hypothesis))
