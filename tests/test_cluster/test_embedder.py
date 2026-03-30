"""Tests for PromptClusterer and clustering types."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from rosettastone.core.types import PromptPair


def _make_pair(prompt: str | list[dict], response: str = "resp") -> PromptPair:
    """Helper to create a PromptPair with minimal boilerplate."""
    return PromptPair(
        prompt=prompt,
        response=response,
        source_model="openai/gpt-4o",
    )


def _make_pairs(prompts: list[str]) -> list[PromptPair]:
    """Helper to create multiple PromptPairs from a list of prompt strings."""
    return [_make_pair(p) for p in prompts]


# ---------------------------------------------------------------------------
# Types tests (4)
# ---------------------------------------------------------------------------


class TestClusterTypes:
    """Tests for PromptCluster and ClusterResult dataclasses."""

    def test_prompt_cluster_has_required_fields(self) -> None:
        """PromptCluster has cluster_id, label, pairs, centroid fields."""
        from rosettastone.cluster.types import PromptCluster

        pc = PromptCluster(cluster_id=0, label="test")
        assert pc.cluster_id == 0
        assert pc.label == "test"
        assert pc.pairs == []
        assert pc.centroid is None

    def test_cluster_result_has_required_fields(self) -> None:
        """ClusterResult has clusters, noise_pairs, n_clusters, silhouette_score."""
        from rosettastone.cluster.types import ClusterResult

        cr = ClusterResult(clusters=[])
        assert cr.clusters == []
        assert cr.noise_pairs == []
        assert cr.n_clusters == 0
        assert cr.silhouette_score is None

    def test_n_clusters_matches_len_clusters(self) -> None:
        """n_clusters should match len(clusters) when set properly."""
        from rosettastone.cluster.types import ClusterResult, PromptCluster

        clusters = [
            PromptCluster(cluster_id=0, label="a"),
            PromptCluster(cluster_id=1, label="b"),
        ]
        cr = ClusterResult(clusters=clusters, n_clusters=2)
        assert cr.n_clusters == len(cr.clusters)

    def test_noise_pairs_is_list_of_prompt_pair(self) -> None:
        """noise_pairs should hold PromptPair instances."""
        from rosettastone.cluster.types import ClusterResult

        pair = _make_pair("noise prompt")
        cr = ClusterResult(clusters=[], noise_pairs=[pair])
        assert len(cr.noise_pairs) == 1
        assert isinstance(cr.noise_pairs[0], PromptPair)


# ---------------------------------------------------------------------------
# Clusterer tests (10)
# ---------------------------------------------------------------------------


class TestClusterer:
    """Tests for the main PromptClusterer.cluster() method."""

    def test_returns_cluster_result_type(self) -> None:
        """cluster() returns a ClusterResult instance."""
        from rosettastone.cluster.embedder import PromptClusterer
        from rosettastone.cluster.types import ClusterResult

        pairs = _make_pairs(["hello", "world", "foo", "bar", "baz"])
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.1],
                [0.0, 1.0],
                [0.1, 1.0],
                [0.5, 0.5],
            ]
        )
        labels = np.array([0, 0, 1, 1, 0])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert isinstance(result, ClusterResult)

    def test_similar_prompts_grouped_together(self) -> None:
        """Prompts with similar embeddings should be in the same cluster."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["cat", "kitten", "dog", "puppy"])
        # Two clear groups: [0,1] near (1,0) and [2,3] near (0,1)
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        # Find which cluster has "cat"
        cat_cluster = None
        kitten_cluster = None
        for c in result.clusters:
            for p in c.pairs:
                if p.prompt == "cat":
                    cat_cluster = c.cluster_id
                if p.prompt == "kitten":
                    kitten_cluster = c.cluster_id
        assert cat_cluster == kitten_cluster

    def test_n_clusters_honored(self) -> None:
        """Specified n_clusters is respected in the result."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["a", "b", "c", "d", "e", "f"])
        embeddings = np.random.default_rng(42).random((6, 4))
        labels = np.array([0, 0, 1, 1, 2, 2])

        clusterer = PromptClusterer(n_clusters=3)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert result.n_clusters == 3

    def test_auto_detection_when_n_clusters_none(self) -> None:
        """When n_clusters is None, defaults to min(5, n_pairs)."""
        from rosettastone.cluster.embedder import PromptClusterer

        clusterer = PromptClusterer(n_clusters=None)
        assert clusterer._n_clusters is None

        # With 3 pairs, _fit_clusters should use n=min(5,3)=3
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        with patch("rosettastone.cluster.embedder.KMeans") as mock_kmeans:
            mock_kmeans.return_value.fit_predict.return_value = np.array([0, 1, 2])
            clusterer._fit_clusters(embeddings)
            mock_kmeans.assert_called_once_with(n_clusters=3, n_init="auto")

    def test_all_pairs_assigned_kmeans(self) -> None:
        """In kmeans mode, every pair is assigned to a cluster (no noise)."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["a", "b", "c", "d", "e"])
        embeddings = np.random.default_rng(42).random((5, 4))
        labels = np.array([0, 1, 0, 1, 0])

        clusterer = PromptClusterer(n_clusters=2, method="kmeans")
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        total_in_clusters = sum(len(c.pairs) for c in result.clusters)
        assert total_in_clusters == len(pairs)
        assert len(result.noise_pairs) == 0

    def test_labels_are_nonempty_strings(self) -> None:
        """Each cluster should have a non-empty label."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["hello world", "hello there", "goodbye now"])
        embeddings = np.random.default_rng(42).random((3, 4))
        labels = np.array([0, 0, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        for c in result.clusters:
            assert isinstance(c.label, str)
            assert len(c.label) > 0

    def test_sequential_cluster_ids(self) -> None:
        """Cluster IDs should be sequential starting from 0."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["a", "b", "c", "d", "e", "f"])
        embeddings = np.random.default_rng(42).random((6, 4))
        labels = np.array([0, 0, 1, 1, 2, 2])

        clusterer = PromptClusterer(n_clusters=3)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        ids = sorted([c.cluster_id for c in result.clusters])
        assert ids == list(range(len(result.clusters)))

    def test_centroid_computed_for_each_cluster(self) -> None:
        """Each cluster should have a non-None centroid."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["a", "b", "c", "d"])
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        for c in result.clusters:
            assert c.centroid is not None
            assert len(c.centroid) == 2  # same dims as embeddings

    def test_silhouette_score_computed_with_two_plus_clusters(self) -> None:
        """Silhouette score is computed when there are 2+ clusters."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["a", "b", "c", "d"])
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
            patch(
                "rosettastone.cluster.embedder.silhouette_score",
                return_value=0.85,
            ),
        ):
            result = clusterer.cluster(pairs)

        assert result.silhouette_score is not None
        assert isinstance(result.silhouette_score, float)

    def test_min_cluster_size_stored(self) -> None:
        """min_cluster_size parameter is stored on the instance."""
        from rosettastone.cluster.embedder import PromptClusterer

        clusterer = PromptClusterer(min_cluster_size=10)
        assert clusterer._min_cluster_size == 10


# ---------------------------------------------------------------------------
# Edge cases (8)
# ---------------------------------------------------------------------------


class TestClustererEdgeCases:
    """Edge-case tests for PromptClusterer."""

    def test_single_pair_single_cluster(self) -> None:
        """A single pair should result in one cluster."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["only one"])
        embeddings = np.array([[0.5, 0.5]])
        labels = np.array([0])

        clusterer = PromptClusterer(n_clusters=1)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert result.n_clusters == 1
        assert len(result.clusters[0].pairs) == 1

    def test_two_pairs_works(self) -> None:
        """Two pairs should cluster without error."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["first", "second"])
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert result.n_clusters == 2

    def test_identical_prompts_same_cluster(self) -> None:
        """Identical prompts should end up in the same cluster."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["same", "same", "different"])
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        labels = np.array([0, 0, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        # Both "same" pairs in same cluster
        for c in result.clusters:
            same_count = sum(1 for p in c.pairs if p.prompt == "same")
            assert same_count in (0, 2)  # either both or neither

    def test_empty_list_returns_empty_result(self) -> None:
        """Empty pair list returns empty ClusterResult."""
        from rosettastone.cluster.embedder import PromptClusterer

        clusterer = PromptClusterer()
        result = clusterer.cluster([])

        assert result.n_clusters == 0
        assert result.clusters == []

    def test_n_clusters_capped_to_n_pairs(self) -> None:
        """n_clusters > n_pairs should be capped to n_pairs."""
        from rosettastone.cluster.embedder import PromptClusterer

        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

        clusterer = PromptClusterer(n_clusters=100)
        with patch.object(clusterer, "_embed_prompts", return_value=embeddings):
            with patch("rosettastone.cluster.embedder.KMeans") as mock_kmeans:
                mock_kmeans.return_value.fit_predict.return_value = np.array([0, 1])
                clusterer._fit_clusters(embeddings)
                # Should cap to 2 (n_pairs), not 100
                called_n = mock_kmeans.call_args[1]["n_clusters"]
                assert called_n == 2

    def test_different_topics_cluster_separately(self) -> None:
        """Distinct topics should end up in different clusters."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(
            [
                "explain quantum physics",
                "quantum mechanics intro",
                "best chocolate cake recipe",
                "how to bake brownies",
            ]
        )
        # Two distinct regions
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert result.n_clusters == 2
        cluster_prompts = [{p.prompt for p in c.pairs} for c in result.clusters]
        # Physics and baking in separate clusters
        for prompts_set in cluster_prompts:
            has_physics = any("quantum" in p for p in prompts_set)
            has_baking = any("cake" in p or "brownie" in p for p in prompts_set)
            assert not (has_physics and has_baking)

    def test_hdbscan_produces_noise_pairs(self) -> None:
        """HDBSCAN mode can produce noise_pairs for outlier labels (-1)."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["a", "b", "c", "outlier"])
        embeddings = np.random.default_rng(42).random((4, 4))
        # Label -1 means noise/outlier in HDBSCAN
        labels = np.array([0, 0, 0, -1])

        clusterer = PromptClusterer(method="hdbscan")
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert len(result.noise_pairs) == 1
        assert result.noise_pairs[0].prompt == "outlier"

    def test_short_prompts_work(self) -> None:
        """Very short prompts (1-2 words) should cluster without error."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["hi", "ok", "no", "go", "up"])
        embeddings = np.random.default_rng(42).random((5, 4))
        labels = np.array([0, 0, 1, 1, 0])

        clusterer = PromptClusterer(n_clusters=2)
        with (
            patch.object(clusterer, "_embed_prompts", return_value=embeddings),
            patch.object(clusterer, "_fit_clusters", return_value=labels),
        ):
            result = clusterer.cluster(pairs)

        assert result.n_clusters == 2


# ---------------------------------------------------------------------------
# Embedding tests (6)
# ---------------------------------------------------------------------------


class TestEmbedding:
    """Tests for _embed_prompts and _extract_prompt_text."""

    def test_embed_returns_ndarray(self) -> None:
        """_embed_prompts returns a numpy ndarray."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["hello", "world"])
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        clusterer = PromptClusterer()
        with patch(
            "rosettastone.cluster.embedder.SentenceTransformer",
            return_value=mock_model,
        ):
            result = clusterer._embed_prompts(pairs)

        assert isinstance(result, np.ndarray)

    def test_string_prompts_embedded(self) -> None:
        """String prompts are passed directly to the embedder."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["hello world"])
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        clusterer = PromptClusterer()
        with patch(
            "rosettastone.cluster.embedder.SentenceTransformer",
            return_value=mock_model,
        ):
            clusterer._embed_prompts(pairs)

        mock_model.encode.assert_called_once_with(["hello world"])

    def test_list_dict_prompts_extracted(self) -> None:
        """List[dict] prompts have content extracted and joined."""
        from rosettastone.cluster.embedder import PromptClusterer

        pair = _make_pair(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Tell me a joke."},
            ]
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        clusterer = PromptClusterer()
        with patch(
            "rosettastone.cluster.embedder.SentenceTransformer",
            return_value=mock_model,
        ):
            clusterer._embed_prompts([pair])

        called_texts = mock_model.encode.call_args[0][0]
        assert "You are helpful." in called_texts[0]
        assert "Tell me a joke." in called_texts[0]

    def test_mixed_prompt_formats_work(self) -> None:
        """Mixed string and list[dict] prompts both get embedded."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = [
            _make_pair("plain text prompt"),
            _make_pair([{"role": "user", "content": "structured prompt"}]),
        ]
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        clusterer = PromptClusterer()
        with patch(
            "rosettastone.cluster.embedder.SentenceTransformer",
            return_value=mock_model,
        ):
            result = clusterer._embed_prompts(pairs)

        assert result.shape[0] == 2

    def test_tfidf_fallback_when_no_sentence_transformers(self) -> None:
        """TF-IDF fallback is used when sentence_transformers is unavailable."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["hello world", "foo bar baz"])
        tfidf_result = np.array([[0.5, 0.5], [0.3, 0.7]])

        clusterer = PromptClusterer()
        with patch.object(clusterer, "_tfidf_embed", return_value=tfidf_result) as mock_tfidf:
            # Force ImportError on sentence_transformers
            with patch(
                "rosettastone.cluster.embedder.SentenceTransformer",
                side_effect=ImportError("no module"),
            ):
                result = clusterer._embed_prompts(pairs)

        mock_tfidf.assert_called_once()
        assert isinstance(result, np.ndarray)

    def test_consistent_embedding_dimensions(self) -> None:
        """All embeddings have the same dimensionality."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["short", "a much longer prompt with many words"])
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )

        clusterer = PromptClusterer()
        with patch(
            "rosettastone.cluster.embedder.SentenceTransformer",
            return_value=mock_model,
        ):
            result = clusterer._embed_prompts(pairs)

        assert result.shape[0] == 2
        assert result.shape[1] == 3  # all same dim


# ---------------------------------------------------------------------------
# Auto-label tests (4)
# ---------------------------------------------------------------------------


class TestAutoLabel:
    """Tests for _auto_label cluster labeling."""

    def test_returns_common_words(self) -> None:
        """Label contains common words from the cluster prompts."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(
            [
                "how to train a neural network",
                "training deep neural networks",
                "neural network training tips",
            ]
        )

        clusterer = PromptClusterer()
        label = clusterer._auto_label(pairs)

        # "neural" and "network" or "training" should appear
        label_lower = label.lower()
        assert "neural" in label_lower or "network" in label_lower

    def test_returns_string_type(self) -> None:
        """_auto_label returns a string."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["hello world"])
        clusterer = PromptClusterer()
        label = clusterer._auto_label(pairs)

        assert isinstance(label, str)

    def test_max_length_truncated(self) -> None:
        """Label is truncated to at most 50 characters."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(
            [
                "superlongword " * 20,
                "superlongword " * 20,
            ]
        )

        clusterer = PromptClusterer()
        label = clusterer._auto_label(pairs)

        assert len(label) <= 50

    def test_single_pair_meaningful_label(self) -> None:
        """A single-pair cluster gets a non-empty label."""
        from rosettastone.cluster.embedder import PromptClusterer

        pairs = _make_pairs(["explain quantum computing basics"])
        clusterer = PromptClusterer()
        label = clusterer._auto_label(pairs)

        assert isinstance(label, str)
        assert len(label) > 0
