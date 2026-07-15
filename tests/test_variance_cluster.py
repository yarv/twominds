"""Tests for embedding clustering + judge-agreement metrics."""

import numpy as np

from coherence_variance import cluster as C


def _norm(mat):
    mat = np.asarray(mat, dtype=float)
    return mat / np.linalg.norm(mat, axis=1, keepdims=True)


def test_two_clean_clusters():
    emb = _norm([[1, 0], [0.99, 0.01], [0, 1], [0.02, 0.98]])
    labels = C.cluster_responses(emb, threshold=0.15)
    # two pairs -> two clusters
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_single_cluster_when_all_similar():
    emb = _norm([[1, 0.01], [1, 0.0], [0.99, 0.02]])
    labels = C.cluster_responses(emb, threshold=0.15)
    assert len(set(labels)) == 1


def test_edge_cases_len_0_and_1():
    assert C.cluster_responses(np.zeros((0, 4))) == []
    assert C.cluster_responses(np.ones((1, 4))) == [0]


def test_agreement_perfect_and_independent():
    perfect = C.agreement([0, 0, 1, 1], [1, 1, 0, 0])  # relabeled but same partition
    assert perfect["ari"] == 1.0 and perfect["nmi"] == 1.0
    # single element -> trivially agrees
    assert C.agreement([0], [0])["ari"] == 1.0


def test_mean_pairwise_cosine_distance():
    same = _norm([[1, 0], [1, 0]])
    assert C.mean_pairwise_cosine_distance(same) < 1e-6
    orth = _norm([[1, 0], [0, 1]])
    assert abs(C.mean_pairwise_cosine_distance(orth) - 1.0) < 1e-6
    assert C.mean_pairwise_cosine_distance(np.ones((1, 3))) == 0.0
