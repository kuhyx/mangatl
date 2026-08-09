"""Reference-free quality estimation for candidate translations.

BLEU is a bad fit for manga: the Manga109/OpenMantra work found context-aware
systems that humans preferred scored *lower* BLEU than sentence-level ones,
because good manga translation legitimately reorders and rewrites. Worse, at
inference time there is no reference translation to score against at all.

So this module scores candidates the way a human QC pass approximates: round
trip the candidate back to the source language and measure how much of the
original survived, using chrF (character n-gram F-score), which is far more
robust than BLEU on short, morphologically dense Japanese lines.
"""

from __future__ import annotations

from collections import Counter

DEFAULT_MAX_N = 6
DEFAULT_BETA = 2.0


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Count character n-grams of size ``n`` in ``text`` ignoring whitespace."""
    squashed = "".join(text.split())
    if len(squashed) < n:
        return Counter()
    return Counter(squashed[i : i + n] for i in range(len(squashed) - n + 1))


def _f_score(precision: float, recall: float, beta: float) -> float:
    """Combine precision and recall into an F-beta score."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)


def chrf(
    hypothesis: str, reference: str, *, max_n: int = DEFAULT_MAX_N, beta: float = DEFAULT_BETA
) -> float:
    """Compute the chrF score between two strings.

    Args:
        hypothesis: Candidate string.
        reference: String to compare against.
        max_n: Largest character n-gram order to consider.
        beta: Recall weight; the standard chrF setting is ``2.0``.

    Returns:
        A score in ``[0, 1]``, where ``1`` means identical non-whitespace
        character content at every order.

    Raises:
        ValueError: If ``max_n`` is not positive.
    """
    if max_n < 1:
        msg = f"max_n must be >= 1, got {max_n}"
        raise ValueError(msg)
    scores: list[float] = []
    for n in range(1, max_n + 1):
        hyp = _char_ngrams(hypothesis, n)
        ref = _char_ngrams(reference, n)
        if not hyp and not ref:
            continue
        overlap = sum((hyp & ref).values())
        precision = overlap / sum(hyp.values()) if hyp else 0.0
        recall = overlap / sum(ref.values()) if ref else 0.0
        scores.append(_f_score(precision, recall, beta))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def length_sanity(source: str, target: str, *, low: float = 0.25, high: float = 4.0) -> float:
    """Penalise target lines that are absurdly long or short for their source.

    A collapsed or runaway line is the most common local-LLM failure mode and
    is trivially detectable without a reference.

    Args:
        source: Source-language line.
        target: Candidate translation.
        low: Ratio below which the target is considered truncated.
        high: Ratio above which the target is considered runaway.

    Returns:
        ``1.0`` when the length ratio is plausible, degrading toward ``0.0``
        as it leaves the ``[low, high]`` band. Empty source scores ``0.0``
        unless the target is empty too, which scores ``1.0``.
    """
    src = len("".join(source.split()))
    tgt = len("".join(target.split()))
    if src == 0:
        return 1.0 if tgt == 0 else 0.0
    if tgt == 0:
        return 0.0
    ratio = tgt / src
    if low <= ratio <= high:
        return 1.0
    if ratio < low:
        return max(0.0, ratio / low)
    return max(0.0, high / ratio)


def score_candidate(source: str, target: str, back_translation: str) -> float:
    """Score one candidate translation without a reference.

    Args:
        source: Original source-language line.
        target: Candidate translation.
        back_translation: The candidate translated back to the source
            language by the same model.

    Returns:
        A score in ``[0, 1]`` combining round-trip fidelity with length
        sanity. Higher is better.
    """
    fidelity = chrf(back_translation, source)
    sanity = length_sanity(source, target)
    return 0.75 * fidelity + 0.25 * sanity


def best_candidate(source: str, candidates: list[tuple[str, str]]) -> tuple[str, float]:
    """Pick the highest-scoring candidate for one line.

    Args:
        source: Original source-language line.
        candidates: ``(target, back_translation)`` pairs.

    Returns:
        The winning ``(target, score)`` pair.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        msg = "need at least one candidate to choose from"
        raise ValueError(msg)
    scored = [(target, score_candidate(source, target, back)) for target, back in candidates]
    return max(scored, key=lambda pair: pair[1])
