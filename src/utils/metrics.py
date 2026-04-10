"""
Evaluation metrics for LLMicro
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate precision for read-level classification.

    Precision = TP / (TP + FP)

    Args:
        y_true: True taxonomy IDs
        y_pred: Predicted taxonomy IDs

    Returns:
        Precision score (0-1)
    """
    # True positives: correctly classified reads
    tp = np.sum((y_pred == y_true) & (y_true != 0))

    # False positives: incorrectly classified reads
    fp = np.sum((y_pred != y_true) & (y_pred != 0))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate recall for read-level classification.

    Recall = TP / (TP + FN)

    Args:
        y_true: True taxonomy IDs
        y_pred: Predicted taxonomy IDs

    Returns:
        Recall score (0-1)
    """
    # True positives: correctly classified reads
    tp = np.sum((y_pred == y_true) & (y_true != 0))

    # False negatives: missed true reads
    fn = np.sum((y_pred != y_true) & (y_true != 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def calculate_f1(precision: float, recall: float) -> float:
    """
    Calculate F1-score from precision and recall.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1-score (0-1)
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_read_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all read-level metrics.

    Returns:
        Dictionary with precision, recall, f1
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(precision, recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_l1_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate L1-norm error for abundance estimation.

    L1 = sum(|true_abundance - predicted_abundance|)

    Args:
        y_true: True abundances
        y_pred: Predicted abundances

    Returns:
        L1-norm error
    """
    return float(np.sum(np.abs(y_true - y_pred)))


def calculate_l2_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate L2 distance for abundance estimation.

    L2 = sqrt(sum((true_abundance - predicted_abundance)^2))

    Args:
        y_true: True abundances
        y_pred: Predicted abundances

    Returns:
        L2 distance
    """
    return float(np.sqrt(np.sum((y_true - y_pred) ** 2)))


def calculate_profiling_f1(true_taxa: set, pred_taxa: set, true_abundances: Dict, pred_abundances: Dict,
                           threshold: float = 0.0001) -> float:
    """
    Calculate profiling F1-score for species detection.

    Args:
        true_taxa: Set of true taxonomy IDs
        pred_taxa: Set of predicted taxonomy IDs
        true_abundances: Dictionary of true abundances
        pred_abundances: Dictionary of predicted abundances
        threshold: Minimum abundance threshold for considering a taxon detected

    Returns:
        Profiling F1-score
    """
    # Filter by abundance threshold
    true_detected = {t for t in true_taxa if true_abundances.get(t, 0) >= threshold}
    pred_detected = {t for t in pred_taxa if pred_abundances.get(t, 0) >= threshold}

    # Calculate precision and recall
    if len(pred_detected) == 0:
        precision = 0.0
    else:
        precision = len(true_detected & pred_detected) / len(pred_detected)

    if len(true_detected) == 0:
        recall = 0.0
    else:
        recall = len(true_detected & pred_detected) / len(true_detected)

    return calculate_f1(precision, recall)


def count_false_positives(pred_abundances: Dict, true_taxa: set,
                          threshold: float = 0.0001) -> int:
    """
    Count false positive taxa above abundance threshold.

    Args:
        pred_abundances: Dictionary of predicted abundances
        true_taxa: Set of true taxonomy IDs
        threshold: Minimum abundance threshold

    Returns:
        Number of false positive taxa
    """
    fp_count = 0
    for taxon, abundance in pred_abundances.items():
        if taxon not in true_taxa and abundance >= threshold:
            fp_count += 1
    return fp_count


def calculate_bray_curtis_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Bray-Curtis dissimilarity between true and predicted abundances.

    BC = sum(|true - pred|) / sum(true + pred)

    Returns:
        Bray-Curtis dissimilarity (0-1)
    """
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(y_true + y_pred)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def calculate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                    taxonomy_levels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate classification report at different taxonomy levels.

    Args:
        y_true: True taxonomy IDs
        y_pred: Predicted taxonomy IDs
        taxonomy_levels: List of taxonomy level names

    Returns:
        DataFrame with metrics per level
    """
    metrics = calculate_read_level_metrics(y_true, y_pred)

    report = pd.DataFrame([
        {'level': 'overall', **metrics}
    ])

    return report


def calculate_auROC(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate area under ROC curve.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores

    Returns:
        AUROC score
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_scores))


def calculate_auPRC(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate area under precision-recall curve.

    Args:
        y_true: True binary labels
        y_scores: Prediction scores

    Returns:
        AUPRC score
    """
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_scores))


def statistical_test(group1: np.ndarray, group2: np.ndarray,
                     test_type: str = 'wilcoxon') -> Tuple[float, float]:
    """
    Perform statistical test between two groups.

    Args:
        group1: First group values
        group2: Second group values
        test_type: 'wilcoxon' for paired, 'mannwhitney' for unpaired

    Returns:
        Tuple of (statistic, p-value)
    """
    if test_type == 'wilcoxon':
        result = stats.wilcoxon(group1, group2)
    elif test_type == 'mannwhitney':
        result = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return result.statistic, result.pvalue


def format_significance(p_value: float) -> str:
    """
    Format p-value with significance stars.

    Returns:
        String with significance notation
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    else:
        return "ns"
