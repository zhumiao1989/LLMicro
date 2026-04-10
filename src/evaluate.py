"""
Evaluation Module for LLMicro

Computes read-level and profile-level metrics for metagenomic classification results.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from loguru import logger

from .utils.io import load_config, parse_kraken2_output, parse_centrifuge_output, parse_pathseq_output
from .utils.metrics import (
    calculate_precision, calculate_recall, calculate_f1,
    calculate_l1_error, calculate_l2_distance,
    calculate_profiling_f1, count_false_positives,
    calculate_bray_curtis_divergence, statistical_test, format_significance
)


@dataclass
class ReadLevelMetrics:
    """Read-level classification metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0
    n_total_reads: int = 0


@dataclass
class ProfileLevelMetrics:
    """Profile-level abundance estimation metrics."""
    profiling_f1: float = 0.0
    l1_norm_error: float = 0.0
    l2_distance: float = 0.0
    bray_curtis: float = 0.0
    n_species_detected: int = 0
    n_species_true: int = 0


@dataclass
class FalsePositiveMetrics:
    """False positive metrics at different abundance thresholds."""
    fp_above_0_001: int = 0  # > 0.001%
    fp_above_0_01: int = 0   # > 0.01%
    fp_above_0_1: int = 0    # > 0.1%


@dataclass
class EvaluationResult:
    """Complete evaluation result for a sample."""
    sample_id: str
    tool: str
    parameter_mode: str  # 'default' or 'llm'
    read_level: ReadLevelMetrics = field(default_factory=ReadLevelMetrics)
    profile_level: ProfileLevelMetrics = field(default_factory=ProfileLevelMetrics)
    false_positives: FalsePositiveMetrics = field(default_factory=FalsePositiveMetrics)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """
    Evaluator for metagenomic classification results.

    Supports evaluation of:
    - Kraken2
    - Centrifuge
    - PathSeq

    Metrics:
    - Read-level: Precision, Recall, F1-score
    - Profile-level: Profiling F1, L1/L2 error, Bray-Curtis
    - False positives at multiple abundance thresholds
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        if config_path:
            self.config = load_config(config_path)

        logger.info("Initialized Evaluator")

    def evaluate_read_level(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame
    ) -> ReadLevelMetrics:
        """
        Evaluate read-level classification accuracy.

        Args:
            predictions: DataFrame with read_id and predicted taxonomy_id
            ground_truth: DataFrame with read_id and true taxonomy_id

        Returns:
            ReadLevelMetrics object
        """
        # Merge predictions with ground truth
        merged = pd.merge(
            ground_truth[['read_id', 'taxonomy_id']],
            predictions[['read_id', 'taxonomy_id']],
            on='read_id',
            suffixes=('_true', '_pred')
        )

        y_true = merged['taxonomy_id_true'].values
        y_pred = merged['taxonomy_id_pred'].values

        # Calculate metrics
        precision = calculate_precision(y_true, y_pred)
        recall = calculate_recall(y_true, y_pred)
        f1 = calculate_f1(precision, recall)

        # Count TP, FP, FN
        tp = np.sum((y_pred == y_true) & (y_true != 0))
        fp = np.sum((y_pred != y_true) & (y_pred != 0))
        fn = np.sum((y_pred != y_true) & (y_true != 0))

        return ReadLevelMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            n_true_positives=int(tp),
            n_false_positives=int(fp),
            n_false_negatives=int(fn),
            n_total_reads=len(merged)
        )

    def evaluate_profile_level(
        self,
        predicted_abundances: Dict[str, float],
        true_abundances: Dict[str, float],
        abundance_threshold: float = 0.0001
    ) -> ProfileLevelMetrics:
        """
        Evaluate profile-level abundance estimation.

        Args:
            predicted_abundances: Dictionary of predicted abundances {taxonomy_id: abundance}
            true_abundances: Dictionary of true abundances {taxonomy_id: abundance}
            abundance_threshold: Minimum abundance for considering a taxon detected

        Returns:
            ProfileLevelMetrics object
        """
        # Get all taxa
        all_taxa = set(predicted_abundances.keys()) | set(true_abundances.keys())

        # Create aligned arrays
        taxa_list = list(all_taxa)
        y_true = np.array([true_abundances.get(t, 0) for t in taxa_list])
        y_pred = np.array([predicted_abundances.get(t, 0) for t in taxa_list])

        # Calculate profiling F1
        true_taxa = set(t for t, a in true_abundances.items() if a >= abundance_threshold)
        pred_taxa = set(t for t, a in predicted_abundances.items() if a >= abundance_threshold)
        profiling_f1 = calculate_profiling_f1(true_taxa, pred_taxa, true_abundances, predicted_abundances, abundance_threshold)

        # Calculate L1 and L2 errors
        l1_norm = calculate_l1_error(y_true, y_pred)
        l2_dist = calculate_l2_distance(y_true, y_pred)

        # Calculate Bray-Curtis dissimilarity
        bray_curtis = calculate_bray_curtis_divergence(y_true, y_pred)

        return ProfileLevelMetrics(
            profiling_f1=profiling_f1,
            l1_norm_error=l1_norm,
            l2_distance=l2_dist,
            bray_curtis=bray_curtis,
            n_species_detected=len(pred_taxa),
            n_species_true=len(true_taxa)
        )

    def evaluate_false_positives(
        self,
        predicted_abundances: Dict[str, float],
        true_taxa: set
    ) -> FalsePositiveMetrics:
        """
        Count false positives at different abundance thresholds.

        Args:
            predicted_abundances: Dictionary of predicted abundances
            true_taxa: Set of true taxonomy IDs

        Returns:
            FalsePositiveMetrics object
        """
        thresholds = [0.00001, 0.0001, 0.001]  # 0.001%, 0.01%, 0.1%

        fp_counts = {}
        for threshold in thresholds:
            fp_counts[f"fp_above_{threshold*100:.3g}"] = count_false_positives(
                predicted_abundances, true_taxa, threshold
            )

        return FalsePositiveMetrics(
            fp_above_0_001=fp_counts.get('fp_above_0.001%', 0),
            fp_above_0_01=fp_counts.get('fp_above_0.01%', 0),
            fp_above_0_1=fp_counts.get('fp_above_0.1%', 0)
        )

    def evaluate_sample(
        self,
        sample_id: str,
        tool: str,
        parameter_mode: str,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        predicted_abundances: Dict[str, float],
        true_abundances: Dict[str, float]
    ) -> EvaluationResult:
        """
        Perform complete evaluation for a sample.

        Args:
            sample_id: Sample identifier
            tool: Classification tool name
            parameter_mode: 'default' or 'llm'
            predictions: Read-level predictions
            ground_truth: Ground truth data
            predicted_abundances: Predicted abundance profile
            true_abundances: True abundance profile

        Returns:
            EvaluationResult object
        """
        # Read-level evaluation
        read_level = self.evaluate_read_level(predictions, ground_truth)

        # Profile-level evaluation
        profile_level = self.evaluate_profile_level(predicted_abundances, true_abundances)

        # False positive evaluation
        true_taxa = set(true_abundances.keys())
        false_positives = self.evaluate_false_positives(predicted_abundances, true_taxa)

        return EvaluationResult(
            sample_id=sample_id,
            tool=tool,
            parameter_mode=parameter_mode,
            read_level=read_level,
            profile_level=profile_level,
            false_positives=false_positives
        )

    def evaluate_dataset(
        self,
        results_dir: str,
        ground_truth_dir: str,
        tool: str
    ) -> pd.DataFrame:
        """
        Evaluate all samples in a dataset.

        Args:
            results_dir: Directory with classification results
            ground_truth_dir: Directory with ground truth files
            tool: Classification tool name

        Returns:
            DataFrame with all metrics
        """
        results_path = Path(results_dir)
        gt_path = Path(ground_truth_dir)

        all_results = []

        # Find all result files
        result_files = list(results_path.glob('*_classification.tsv')) + \
                       list(results_path.glob('*_report.tsv'))

        for result_file in result_files:
            sample_id = result_file.stem.replace('_classification', '').replace('_report', '')

            # Load ground truth
            gt_file = gt_path / f"{sample_id}_ground_truth.tsv"
            if not gt_file.exists():
                logger.warning(f"Ground truth not found for {sample_id}")
                continue

            ground_truth = pd.read_csv(gt_file, sep='\t')

            # Load predictions (format depends on tool)
            if tool == 'kraken2':
                predictions = parse_kraken2_output(result_file)
            elif tool == 'centrifuge':
                predictions = parse_centrifuge_output(result_file)
            elif tool == 'pathseq':
                predictions = parse_pathseq_output(result_file)
            else:
                predictions = pd.read_csv(result_file, sep='\t')

            # Extract abundances
            true_abundances = dict(zip(
                ground_truth['taxonomy_id'].unique(),
                ground_truth.groupby('taxonomy_id').size() / len(ground_truth)
            ))

            predicted_abundances = dict(zip(
                predictions['taxonomy_id'] if 'taxonomy_id' in predictions.columns else predictions['name'],
                predictions.get('abundance', predictions.get('percentage', []))
            ))

            # Determine parameter mode from filename or metadata
            parameter_mode = 'default'  # Would be determined from metadata
            if 'llm' in result_file.stem.lower() or 'recommended' in result_file.stem.lower():
                parameter_mode = 'llm'

            # Evaluate
            result = self.evaluate_sample(
                sample_id=sample_id,
                tool=tool,
                parameter_mode=parameter_mode,
                predictions=predictions[['read_id', 'taxonomy_id']] if 'read_id' in predictions.columns else predictions,
                ground_truth=ground_truth,
                predicted_abundances=predicted_abundances,
                true_abundances=true_abundances
            )

            all_results.append(result)

        # Convert to DataFrame
        return self.results_to_dataframe(all_results)

    def results_to_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """Convert evaluation results to DataFrame."""
        rows = []

        for result in results:
            row = {
                'sample_id': result.sample_id,
                'tool': result.tool,
                'parameter_mode': result.parameter_mode,
                'precision': result.read_level.precision,
                'recall': result.read_level.recall,
                'read_f1': result.read_level.f1,
                'profiling_f1': result.profile_level.profiling_f1,
                'l1_norm_error': result.profile_level.l1_norm_error,
                'l2_distance': result.profile_level.l2_distance,
                'bray_curtis': result.profile_level.bray_curtis,
                'fp_above_0.001%': result.false_positives.fp_above_0_001,
                'fp_above_0.01%': result.false_positives.fp_above_0_01,
                'fp_above_0.1%': result.false_positives.fp_above_0_1,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def compare_parameter_modes(
        self,
        results: pd.DataFrame,
        test_type: str = 'wilcoxon'
    ) -> pd.DataFrame:
        """
        Compare default vs LLM-recommended parameters.

        Args:
            results: DataFrame with evaluation results
            test_type: Statistical test to use

        Returns:
            DataFrame with comparison statistics
        """
        comparisons = []

        for tool in results['tool'].unique():
            tool_results = results[results['tool'] == tool]

            default = tool_results[tool_results['parameter_mode'] == 'default']
            llm = tool_results[tool_results['parameter_mode'] == 'llm']

            if len(default) == 0 or len(llm) == 0:
                continue

            metrics_to_compare = [
                'precision', 'recall', 'read_f1',
                'profiling_f1', 'l1_norm_error', 'l2_distance',
                'fp_above_0.01%'
            ]

            for metric in metrics_to_compare:
                if metric not in default.columns or metric not in llm.columns:
                    continue

                default_vals = default[metric].values
                llm_vals = llm[metric].values

                stat, p_value = statistical_test(llm_vals, default_vals, test_type)

                comparisons.append({
                    'tool': tool,
                    'metric': metric,
                    'default_mean': float(np.mean(default_vals)),
                    'default_std': float(np.std(default_vals)),
                    'llm_mean': float(np.mean(llm_vals)),
                    'llm_std': float(np.std(llm_vals)),
                    'statistic': stat,
                    'p_value': p_value,
                    'significance': format_significance(p_value)
                })

        return pd.DataFrame(comparisons)

    def save_results(
        self,
        results: Union[pd.DataFrame, List[EvaluationResult]],
        output_path: str
    ) -> None:
        """Save evaluation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(results, pd.DataFrame):
            results.to_csv(output_file, index=False)
        else:
            # Convert to dict and save as JSON
            data = []
            for r in results:
                data.append({
                    'sample_id': r.sample_id,
                    'tool': r.tool,
                    'parameter_mode': r.parameter_mode,
                    'read_level': {
                        'precision': r.read_level.precision,
                        'recall': r.read_level.recall,
                        'f1': r.read_level.f1
                    },
                    'profile_level': {
                        'profiling_f1': r.profile_level.profiling_f1,
                        'l1_norm_error': r.profile_level.l1_norm_error,
                        'l2_distance': r.profile_level.l2_distance
                    },
                    'false_positives': {
                        'fp_above_0.001%': r.false_positives.fp_above_0_001,
                        'fp_above_0.01%': r.false_positives.fp_above_0_01,
                        'fp_above_0.1%': r.false_positives.fp_above_0_1
                    }
                })

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

        logger.info(f"Saved evaluation results to {output_path}")


def main():
    """CLI entry point."""
    import click

    @click.command()
    @click.option('--results', '-r', required=True, help='Results directory')
    @click.option('--ground-truth', '-g', required=True, help='Ground truth directory')
    @click.option('--tool', '-t', type=click.Choice(['kraken2', 'centrifuge', 'pathseq']), required=True)
    @click.option('--output', '-o', required=True, help='Output file')
    def evaluate(results, ground_truth, tool, output):
        """Evaluate classification results."""
        evaluator = Evaluator()

        df = evaluator.evaluate_dataset(results, ground_truth, tool)
        evaluator.save_results(df, output)

        click.echo(f"Evaluation complete. Results saved to {output}")
        click.echo(f"\nSummary:")
        click.echo(df.groupby(['tool', 'parameter_mode']).mean()[['precision', 'recall', 'f1']])

    evaluate()


if __name__ == '__main__':
    main()
