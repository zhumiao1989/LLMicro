"""
Tests for Evaluation Module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluate import Evaluator, ReadLevelMetrics, ProfileLevelMetrics
from src.utils.metrics import (
    calculate_precision, calculate_recall, calculate_f1,
    calculate_l1_error, calculate_l2_distance
)


class TestMetrics:
    """Test cases for metric calculation functions."""

    def test_perfect_precision(self):
        """Test precision with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert calculate_precision(y_true, y_pred) == 1.0

    def test_zero_precision(self):
        """Test precision with all wrong predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        assert calculate_precision(y_true, y_pred) == 0.0

    def test_partial_precision(self):
        """Test precision with partial correct predictions."""
        y_true = np.array([1, 2, 3, 0, 0])  # 0 = unclassified
        y_pred = np.array([1, 2, 4, 0, 0])
        # TP = 2 (1, 2), FP = 1 (4)
        precision = 2 / (2 + 1)
        assert abs(calculate_precision(y_true, y_pred) - precision) < 0.001

    def test_perfect_recall(self):
        """Test recall with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert calculate_recall(y_true, y_pred) == 1.0

    def test_zero_recall(self):
        """Test recall with all missed predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([0, 0, 0, 0, 0])  # All unclassified
        assert calculate_recall(y_true, y_pred) == 0.0

    def test_partial_recall(self):
        """Test recall with partial detection."""
        y_true = np.array([1, 2, 3, 0, 0])
        y_pred = np.array([1, 2, 0, 0, 0])
        # TP = 2, FN = 1 (3 missed)
        recall = 2 / (2 + 1)
        assert abs(calculate_recall(y_true, y_pred) - recall) < 0.001

    def test_f1_perfect(self):
        """Test F1 with perfect precision and recall."""
        assert calculate_f1(1.0, 1.0) == 1.0

    def test_f1_zero(self):
        """Test F1 with zero precision and recall."""
        assert calculate_f1(0.0, 0.0) == 0.0

    def test_f1_balanced(self):
        """Test F1 with balanced precision and recall."""
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert calculate_f1(0.5, 0.5) == 0.5

    def test_l1_error_identical(self):
        """Test L1 error with identical arrays."""
        y_true = np.array([0.3, 0.5, 0.2])
        y_pred = np.array([0.3, 0.5, 0.2])
        assert calculate_l1_error(y_true, y_pred) == 0.0

    def test_l1_error_different(self):
        """Test L1 error with different arrays."""
        y_true = np.array([0.5, 0.3, 0.2])
        y_pred = np.array([0.4, 0.4, 0.2])
        # L1 = |0.5-0.4| + |0.3-0.4| + |0.2-0.2| = 0.2
        assert calculate_l1_error(y_true, y_pred) == 0.2

    def test_l2_distance_identical(self):
        """Test L2 distance with identical arrays."""
        y_true = np.array([0.3, 0.5, 0.2])
        y_pred = np.array([0.3, 0.5, 0.2])
        assert calculate_l2_distance(y_true, y_pred) == 0.0

    def test_l2_distance_different(self):
        """Test L2 distance with different arrays."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 1.0])
        # L2 = sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        assert abs(calculate_l2_distance(y_true, y_pred) - np.sqrt(2)) < 0.001


class TestEvaluator:
    """Test cases for Evaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a test evaluator."""
        return Evaluator()

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        # Ground truth: 100 reads
        ground_truth = pd.DataFrame({
            'read_id': [f'read_{i}' for i in range(100)],
            'taxonomy_id': [1] * 50 + [2] * 30 + [3] * 20,
            'abundance': [0.5] * 50 + [0.3] * 30 + [0.2] * 20,
            'is_human': [False] * 100
        })

        # Predictions: 85% accuracy
        predictions = pd.DataFrame({
            'read_id': [f'read_{i}' for i in range(100)],
            'taxonomy_id': [1] * 45 + [2] * 5 +  # First 50 (5 wrong)
                          [1] * 5 + [2] * 25 +  # Next 30 (5 wrong)
                          [3] * 20  # Last 20 (all correct)
        })

        # Abundances
        true_abundances = {1: 0.5, 2: 0.3, 3: 0.2}
        pred_abundances = {1: 0.5, 2: 0.3, 3: 0.2}

        return {
            'ground_truth': ground_truth,
            'predictions': predictions,
            'true_abundances': true_abundances,
            'pred_abundances': pred_abundances
        }

    def test_evaluate_read_level(self, evaluator, sample_data):
        """Test read-level evaluation."""
        metrics = evaluator.evaluate_read_level(
            sample_data['predictions'],
            sample_data['ground_truth']
        )

        assert isinstance(metrics, ReadLevelMetrics)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert metrics.n_total_reads == 100

    def test_evaluate_profile_level(self, evaluator, sample_data):
        """Test profile-level evaluation."""
        metrics = evaluator.evaluate_profile_level(
            sample_data['pred_abundances'],
            sample_data['true_abundances']
        )

        assert isinstance(metrics, ProfileLevelMetrics)
        assert 0 <= metrics.profiling_f1 <= 1
        assert metrics.l1_norm_error >= 0
        assert metrics.l2_distance >= 0

    def test_evaluate_sample(self, evaluator, sample_data):
        """Test complete sample evaluation."""
        result = evaluator.evaluate_sample(
            sample_id='test_sample',
            tool='kraken2',
            parameter_mode='llm',
            predictions=sample_data['predictions'],
            ground_truth=sample_data['ground_truth'],
            predicted_abundances=sample_data['pred_abundances'],
            true_abundances=sample_data['true_abundances']
        )

        assert result.sample_id == 'test_sample'
        assert result.tool == 'kraken2'
        assert result.parameter_mode == 'llm'
        assert result.read_level is not None
        assert result.profile_level is not None

    def test_results_to_dataframe(self, evaluator):
        """Test converting results to DataFrame."""
        from src.evaluate import EvaluationResult, ReadLevelMetrics, ProfileLevelMetrics, FalsePositiveMetrics

        results = [
            EvaluationResult(
                sample_id='sample_1',
                tool='kraken2',
                parameter_mode='default',
                read_level=ReadLevelMetrics(precision=0.8, recall=0.7, f1=0.75),
                profile_level=ProfileLevelMetrics(profiling_f1=0.85, l1_norm_error=0.2),
                false_positives=FalsePositiveMetrics(fp_above_0_01=5)
            ),
            EvaluationResult(
                sample_id='sample_2',
                tool='kraken2',
                parameter_mode='llm',
                read_level=ReadLevelMetrics(precision=0.85, recall=0.72, f1=0.78),
                profile_level=ProfileLevelMetrics(profiling_f1=0.88, l1_norm_error=0.18),
                false_positives=FalsePositiveMetrics(fp_above_0_01=3)
            )
        ]

        df = evaluator.results_to_dataframe(results)

        assert len(df) == 2
        assert 'sample_id' in df.columns
        assert 'precision' in df.columns
        assert 'recall' in df.columns
        assert 'f1' in df.columns

    def test_compare_parameter_modes(self, evaluator):
        """Test comparing parameter modes."""
        # Create sample results
        results = pd.DataFrame([
            {'sample_id': 's1', 'tool': 'kraken2', 'parameter_mode': 'default',
             'precision': 0.8, 'recall': 0.7, 'f1': 0.75, 'l1_norm_error': 0.2, 'fp_above_0.01%': 5},
            {'sample_id': 's2', 'tool': 'kraken2', 'parameter_mode': 'llm',
             'precision': 0.85, 'recall': 0.72, 'f1': 0.78, 'l1_norm_error': 0.18, 'fp_above_0.01%': 3},
        ])

        comparison = evaluator.compare_parameter_modes(results)

        assert len(comparison) > 0
        assert 'tool' in comparison.columns
        assert 'metric' in comparison.columns
        assert 'p_value' in comparison.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
