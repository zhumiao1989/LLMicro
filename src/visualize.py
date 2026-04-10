"""
Visualization Module for LLMicro

Generates publication-ready figures for metagenomic classification benchmarking.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger

# Plotting libraries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


class Visualizer:
    """
    Visualizer for metagenomic classification benchmarking results.

    Generates figures similar to those in the LLMicro paper:
    - Figure 2: Read-level F1 comparison (low complexity)
    - Figure 3: Profiling L1 error comparison (medium complexity)
    - Figure 4: False positive taxa comparison (high complexity)
    - Figure 5: Mock community Recall/Precision comparison
    """

    def __init__(self, output_dir: str = 'results/figures'):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory for saving figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette (matching paper style)
        self.colors = {
            'default': '#CCCCCC',      # Gray for default
            'llm': '#2E75B6',          # Blue for LLM-recommended
            'kraken2': '#4472C4',
            'centrifuge': '#ED7D31',
            'pathseq': '#70AD47',
        }

        logger.info(f"Initialized Visualizer, output dir: {output_dir}")

    def plot_read_level_f1(
        self,
        data: pd.DataFrame,
        output_name: str = 'figure2_read_level_f1.png',
        figsize: Tuple[int, int] = (8, 5),
        dpi: int = 300
    ) -> str:
        """
        Figure 2: Read-level F1-score comparison on low complexity dataset.

        Args:
            data: DataFrame with columns [tool, parameter_mode, f1]
            output_name: Output filename
            figsize: Figure size
            dpi: Resolution

        Returns:
            Path to saved figure
        """
        logger.info(f"Generating Figure 2: Read-level F1 comparison")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Prepare data for plotting
        plot_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values='f1'
        )

        # Bar positions
        x = np.arange(len(plot_data.index))
        width = 0.35

        # Plot bars
        default_bars = ax.bar(
            x - width/2,
            plot_data['default'],
            width,
            label='Default',
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )

        llm_bars = ax.bar(
            x + width/2,
            plot_data['llm'],
            width,
            label='LLMicro (Recommended)',
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        # Labels and title
        ax.set_ylabel('Read-level F1-score')
        ax.set_xlabel('Classification Tool')
        ax.set_title('Low Complexity Dataset (10-30 species)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])
        ax.legend(frameon=False)
        ax.set_ylim(0, 1.0)

        # Add value labels on bars
        for bars in [default_bars, llm_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        # Add significance stars if available
        if 'p_value' in data.columns:
            # Add stars above bars
            pass  # Would need paired data for this

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Figure 2 to {output_path}")
        return str(output_path)

    def plot_profiling_l1_error(
        self,
        data: pd.DataFrame,
        output_name: str = 'figure3_profiling_l1_error.png',
        figsize: Tuple[int, int] = (8, 5),
        dpi: int = 300
    ) -> str:
        """
        Figure 3: Profiling L1-norm error comparison on medium complexity dataset.

        Args:
            data: DataFrame with columns [tool, parameter_mode, l1_norm_error]
            output_name: Output filename
            figsize: Figure size
            dpi: Resolution

        Returns:
            Path to saved figure
        """
        logger.info(f"Generating Figure 3: Profiling L1 error comparison")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Prepare data
        plot_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values='l1_norm_error'
        )

        x = np.arange(len(plot_data.index))
        width = 0.35

        # Plot bars
        default_bars = ax.bar(
            x - width/2,
            plot_data['default'],
            width,
            label='Default',
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )

        llm_bars = ax.bar(
            x + width/2,
            plot_data['llm'],
            width,
            label='LLMicro (Recommended)',
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        # Labels
        ax.set_ylabel('L1-norm Error')
        ax.set_xlabel('Classification Tool')
        ax.set_title('Medium Complexity Dataset (50-150 species)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])
        ax.legend(frameon=False)
        ax.set_ylim(0, max(plot_data.max()) * 1.2)

        # Add value labels
        for bars in [default_bars, llm_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Figure 3 to {output_path}")
        return str(output_path)

    def plot_false_positives(
        self,
        data: pd.DataFrame,
        threshold: str = '0.01%',
        output_name: str = 'figure4_false_positives.png',
        figsize: Tuple[int, int] = (8, 5),
        dpi: int = 300
    ) -> str:
        """
        Figure 4: False positive taxa comparison on high complexity dataset.

        Args:
            data: DataFrame with columns [tool, parameter_mode, fp_count]
            threshold: Abundance threshold (e.g., '0.01%')
            output_name: Output filename
            figsize: Figure size
            dpi: Resolution

        Returns:
            Path to saved figure
        """
        logger.info(f"Generating Figure 4: False positives comparison")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Get the right column name
        fp_col = f'fp_above_{threshold}'
        if fp_col not in data.columns:
            # Try to find similar column
            for col in data.columns:
                if 'fp' in col.lower() and threshold.replace('%', '') in col:
                    fp_col = col
                    break

        # Prepare data
        plot_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values=fp_col
        )

        x = np.arange(len(plot_data.index))
        width = 0.35

        # Plot bars
        default_bars = ax.bar(
            x - width/2,
            plot_data['default'],
            width,
            label='Default',
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )

        llm_bars = ax.bar(
            x + width/2,
            plot_data['llm'],
            width,
            label='LLMicro (Recommended)',
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        # Labels
        ax.set_ylabel(f'False Positive Taxa (> {threshold})')
        ax.set_xlabel('Classification Tool')
        ax.set_title('High Complexity Dataset (200-400 species)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])
        ax.legend(frameon=False)

        # Add value labels
        for bars in [default_bars, llm_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Figure 4 to {output_path}")
        return str(output_path)

    def plot_mock_recall_precision(
        self,
        data: pd.DataFrame,
        output_name: str = 'figure5_mock_recall_precision.png',
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 300
    ) -> str:
        """
        Figure 5: Recall and Precision comparison on mock community dataset.

        Args:
            data: DataFrame with columns [tool, parameter_mode, recall, precision]
            output_name: Output filename
            figsize: Figure size
            dpi: Resolution

        Returns:
            Path to saved figure
        """
        logger.info(f"Generating Figure 5: Mock community Recall/Precision comparison")

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        # Prepare data for Recall
        recall_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values='recall'
        )

        # Prepare data for Precision
        precision_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values='precision'
        )

        x = np.arange(len(recall_data.index))
        width = 0.35

        # Left panel: Recall
        ax1 = axes[0]
        default_recall = ax1.bar(
            x - width/2,
            recall_data['default'],
            width,
            label='Default',
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )
        llm_recall = ax1.bar(
            x + width/2,
            recall_data['llm'],
            width,
            label='LLMicro',
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        ax1.set_ylabel('Recall')
        ax1.set_xlabel('Classification Tool')
        ax1.set_title('Mock Community - Recall')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])
        ax1.set_ylim(0, 1.0)

        # Right panel: Precision
        ax2 = axes[1]
        default_prec = ax2.bar(
            x - width/2,
            precision_data['default'],
            width,
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )
        llm_prec = ax2.bar(
            x + width/2,
            precision_data['llm'],
            width,
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        ax2.set_ylabel('Precision')
        ax2.set_xlabel('Classification Tool')
        ax2.set_title('Mock Community - Precision')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])
        ax2.set_ylim(0, 1.0)

        # Shared legend
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                   ncol=2, frameon=False)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved Figure 5 to {output_path}")
        return str(output_path)

    def plot_resource_usage(
        self,
        data: pd.DataFrame,
        output_name: str = 'resource_usage.png',
        figsize: Tuple[int, int] = (10, 5),
        dpi: int = 300
    ) -> str:
        """
        Plot resource usage comparison (speed and memory).

        Args:
            data: DataFrame with columns [tool, parameter_mode, speed, memory]
            output_name: Output filename
            figsize: Figure size
            dpi: Resolution

        Returns:
            Path to saved figure
        """
        logger.info(f"Generating resource usage comparison")

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

        # Speed data
        speed_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values='speed'
        )

        # Memory data
        memory_data = data.pivot(
            index='tool',
            columns='parameter_mode',
            values='memory'
        )

        x = np.arange(len(speed_data.index))
        width = 0.35

        # Left panel: Speed (M reads/min)
        ax1 = axes[0]
        default_speed = ax1.bar(
            x - width/2,
            speed_data['default'],
            width,
            label='Default',
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )
        llm_speed = ax1.bar(
            x + width/2,
            speed_data['llm'],
            width,
            label='LLMicro',
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        ax1.set_ylabel('Speed (M reads/min)')
        ax1.set_xlabel('Classification Tool')
        ax1.set_title('Classification Speed')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])

        # Right panel: Memory (GB)
        ax2 = axes[1]
        default_mem = ax2.bar(
            x - width/2,
            memory_data['default'],
            width,
            color=self.colors['default'],
            edgecolor='black',
            linewidth=0.5
        )
        llm_mem = ax2.bar(
            x + width/2,
            memory_data['llm'],
            width,
            color=self.colors['llm'],
            edgecolor='black',
            linewidth=0.5
        )

        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_xlabel('Classification Tool')
        ax2.set_title('Peak Memory Usage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Kraken2', 'Centrifuge', 'PathSeq'])

        # Shared legend
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
                   ncol=2, frameon=False)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved resource usage figure to {output_path}")
        return str(output_path)

    def generate_all_figures(
        self,
        metrics_df: pd.DataFrame,
        output_prefix: str = ''
    ) -> Dict[str, str]:
        """
        Generate all standard figures from metrics DataFrame.

        Args:
            metrics_df: DataFrame with all evaluation metrics
            output_prefix: Prefix for output filenames

        Returns:
            Dictionary mapping figure names to file paths
        """
        generated = {}

        # Filter by complexity type if available
        if 'complexity_type' in metrics_df.columns:
            low_complexity = metrics_df[metrics_df['complexity_type'] == 'low']
            medium_complexity = metrics_df[metrics_df['complexity_type'] == 'medium']
            high_complexity = metrics_df[metrics_df['complexity_type'] == 'high']
            mock = metrics_df[metrics_df['complexity_type'] == 'mock']
        else:
            # Assume all data is from same complexity type
            low_complexity = medium_complexity = high_complexity = mock = metrics_df

        # Figure 2: Read-level F1 (low complexity)
        if len(low_complexity) > 0:
            path = self.plot_read_level_f1(low_complexity, f'{output_prefix}figure2_read_level_f1.png')
            generated['figure2'] = path

        # Figure 3: Profiling L1 error (medium complexity)
        if len(medium_complexity) > 0:
            path = self.plot_profiling_l1_error(medium_complexity, f'{output_prefix}figure3_profiling_l1_error.png')
            generated['figure3'] = path

        # Figure 4: False positives (high complexity)
        if len(high_complexity) > 0:
            path = self.plot_false_positives(high_complexity, f'{output_prefix}figure4_false_positives.png')
            generated['figure4'] = path

        # Figure 5: Mock community
        if len(mock) > 0:
            path = self.plot_mock_recall_precision(mock, f'{output_prefix}figure5_mock_recall_precision.png')
            generated['figure5'] = path

        return generated


def main():
    """CLI entry point."""
    import click

    @click.command()
    @click.option('--metrics', '-m', required=True, help='Metrics CSV file')
    @click.option('--output', '-o', default='results/figures', help='Output directory')
    def generate_figures(metrics, output):
        """Generate all figures from metrics file."""
        df = pd.read_csv(metrics)

        visualizer = Visualizer(output_dir=output)
        generated = visualizer.generate_all_figures(df)

        click.echo("Generated figures:")
        for fig_name, path in generated.items():
            click.echo(f"  {fig_name}: {path}")

    generate_figures()


if __name__ == '__main__':
    main()
