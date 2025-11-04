from typing import List, Union


def levenshtein_distance(s1: Union[str, List], s2: Union[str, List]) -> int:
    """
    Calculates the Levenshtein distance between two sequences.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

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
    if not reference:
        return 1.0 if hypothesis else 0.0
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """
    Calculates the Word Error Rate (WER).
    WER = (Substitutions + Insertions + Deletions) / Total number of words in reference
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)
