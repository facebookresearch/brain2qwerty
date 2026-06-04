# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TF-IDF similarity-based data splitter.

This module implements the sentence-level train/val/test splitting strategy
described in Levy et al. (2025). Sentences are clustered by TF-IDF cosine
similarity so that semantically similar sentences always land in the same
split, preventing data leakage through paraphrases or shared vocabulary.

Usage::

    from brain2qwerty.splitter import split_events, check_leakage

    events = split_events(events, ratios=(0.8, 0.1, 0.1), seed=42)
    leaks = check_leakage(events, threshold=0.5)
    assert len(leaks) == 0, f"Found {len(leaks)} leaking sentence pairs"
"""

import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _cluster_sentences(similarity_matrix: np.ndarray, threshold: float = 0.5):
    """Group sentence indices into clusters where all pairwise similarities
    exceed *threshold* (transitive closure)."""
    n = similarity_matrix.shape[0]
    clusters, visited = [], set()
    for i in range(n):
        if i in visited:
            continue
        cluster = {i}
        expanded = True
        while expanded:
            expanded = False
            for idx in list(cluster):
                for j in range(n):
                    if j not in cluster and similarity_matrix[idx, j] > threshold:
                        cluster.add(j)
                        expanded = True
        visited.update(cluster)
        clusters.append(list(cluster))
    return clusters


def split_events(
    events: pd.DataFrame,
    ratios: tuple[float, ...] = (0.8, 0.1, 0.1),
    seed: int | None = None,
    similarity_threshold: float = 0.5,
) -> pd.DataFrame:
    """Assign train/val/test splits at the sentence level.

    Sentences are clustered by TF-IDF cosine similarity so that
    semantically related sentences always fall in the same split. Clusters
    are then allocated greedily to train, val, and test until the target
    ratios (measured in number of Button events) are reached.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain columns ``type`` (with ``"Button"`` rows) and
        ``sentence``.
    ratios : tuple of float
        Target fractions for (train, val, test). Must sum to 1.
    seed : int or None
        Random seed for reproducibility.
    similarity_threshold : float
        TF-IDF cosine similarity above which two sentences are forced
        into the same split.

    Returns
    -------
    pd.DataFrame
        The input frame with an added ``split`` column.
    """
    buttons = events[events["type"] == "Button"]
    unique_sentences = buttons["sentence"].unique()
    random.seed(seed)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(unique_sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)

    clusters = _cluster_sentences(sim_matrix, similarity_threshold)
    random.shuffle(clusters)

    total = len(buttons)
    split_sizes = {
        "train": int(ratios[0] * total),
        "val": int(ratios[1] * total),
        "test": total - int(ratios[0] * total) - int(ratios[1] * total),
    }
    current = {"train": 0, "val": 0, "test": 0}
    sentence_to_split = {}

    for cluster in clusters:
        cluster_sents = [unique_sentences[idx] for idx in cluster]
        cluster_size = len(buttons[buttons["sentence"].isin(cluster_sents)])
        assigned = "test"
        for split in ["train", "val", "test"]:
            if current[split] + cluster_size <= split_sizes[split]:
                current[split] += cluster_size
                assigned = split
                break
        for s in cluster_sents:
            sentence_to_split[s] = assigned

    events["split"] = events["sentence"].map(sentence_to_split)
    return events


def check_leakage(
    events: pd.DataFrame, threshold: float = 0.5
) -> list[tuple[str, str, float]]:
    """Check for cross-split sentence pairs with TF-IDF similarity above *threshold*.

    Returns a list of ``(sentence_a, sentence_b, similarity)`` tuples for
    any pair that appears in different splits. An empty list means no leakage.
    """
    buttons = events[events["type"] == "Button"]
    split_sentences = {}
    for split in ["train", "val", "test"]:
        split_buttons = buttons[buttons["split"] == split]
        split_sentences[split] = split_buttons["sentence"].unique()

    all_sentences = np.concatenate(list(split_sentences.values()))
    if len(all_sentences) == 0:
        return []

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)

    sent_to_index = {s: i for i, s in enumerate(all_sentences)}
    leaks = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for split_a, split_b in pairs:
        for sa in split_sentences[split_a]:
            for sb in split_sentences[split_b]:
                sim = sim_matrix[sent_to_index[sa], sent_to_index[sb]]
                if sim > threshold:
                    leaks.append((sa, sb, float(sim)))
    return leaks
