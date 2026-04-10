"""
Input/Output utilities for LLMicro
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(data: Dict[str, Any], output_path: str, format: str = 'json') -> None:
    """Save results to file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format == 'tsv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_file, sep='\t', index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(output_file, sep='\t', index=False)
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_file, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Results saved to {output_path}")


def parse_kraken2_output(report_path: str) -> pd.DataFrame:
    """
    Parse Kraken2 report file.

    Returns DataFrame with columns:
    - percentage: Percentage of fragments covered by the clade
    - fragments: Number of fragments covered by the clade
    - taxonomic_rank: Rank code (U, R, D, P, C, O, F, G, S)
    - taxonomy_id: NCBI taxonomy ID
    - indented_name: Indented scientific name
    - name: Scientific name (unindented)
    """
    columns = ['percentage', 'fragments', 'taxonomic_rank', 'taxonomy_id', 'indented_name']
    df = pd.read_csv(report_path, sep='\t', header=None, names=columns)

    # Extract unindented name
    df['name'] = df['indented_name'].str.lstrip()

    return df


def parse_centrifuge_output(report_path: str) -> pd.DataFrame:
    """
    Parse Centrifuge report file.

    Returns DataFrame with columns:
    - name: Taxon name
    - taxonomy_id: Taxonomy ID
    - rank: Taxonomic rank
    - genome_size: Genome size
    - num_reads: Number of assigned reads
    - abundance: Relative abundance
    """
    df = pd.read_csv(report_path, sep='\t', comment='#')
    return df


def parse_pathseq_output(profile_path: str) -> pd.DataFrame:
    """
    Parse PathSeq profile output.

    Returns DataFrame with taxonomy and abundance information.
    """
    df = pd.read_csv(profile_path, sep='\t')
    return df


def load_ground_truth(truth_path: str) -> pd.DataFrame:
    """
    Load ground truth data from simulation.

    Expected format: TSV with columns
    - read_id: Read identifier
    - taxonomy_id: True taxonomy ID
    - abundance: True abundance (for profile-level)
    """
    return pd.read_csv(truth_path, sep='\t')


def load_fastq_stats(fastq_path: str) -> Dict[str, Any]:
    """
    Extract statistics from FASTQ file.

    Returns:
    - n_reads: Total number of reads
    - mean_length: Mean read length
    - mean_quality: Mean quality score
    """
    import gzip

    open_func = gzip.open if str(fastq_path).endswith('.gz') else open

    n_reads = 0
    total_length = 0
    total_quality = 0

    with open_func(fastq_path, 'rt') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:  # Sequence line
                total_length += len(line.strip())
                n_reads += 1
            elif i % 4 == 3:  # Quality line
                # Convert Phred+33 quality scores
                qualities = [ord(c) - 33 for c in line.strip()]
                total_quality += sum(qualities)

    return {
        'n_reads': n_reads,
        'mean_length': total_length / n_reads if n_reads > 0 else 0,
        'mean_quality': total_quality / (total_length if total_length > 0 else 1)
    }


def estimate_complexity(reads: pd.DataFrame) -> float:
    """
    Estimate sample complexity from read taxonomy assignments.

    Returns Shannon diversity index.
    """
    if len(reads) == 0:
        return 0.0

    # Count reads per taxon
    taxon_counts = reads['taxonomy_id'].value_counts()

    # Calculate proportions
    proportions = taxon_counts / taxon_counts.sum()

    # Shannon diversity index
    from scipy.stats import entropy
    return float(entropy(proportions, base=2))


def get_sample_features(data_dir: str) -> Dict[str, Any]:
    """
    Extract sample features for LLM input.

    Returns dictionary with:
    - sequencing_depth: Total number of reads
    - mean_read_length: Average read length
    - mean_quality: Average quality score
    - estimated_complexity: Shannon diversity estimate
    - host_contamination: Estimated host read proportion
    """
    # Find FASTQ files
    fastq_files = list(Path(data_dir).glob('*.fastq')) + list(Path(data_dir).glob('*.fq'))
    if not fastq_files:
        fastq_files = list(Path(data_dir).glob('*.fastq.gz')) + list(Path(data_dir).glob('*.fq.gz'))

    if not fastq_files:
        logger.warning(f"No FASTQ files found in {data_dir}")
        return {}

    # Aggregate stats from all files
    total_reads = 0
    total_length = 0
    total_quality = 0

    for fq in fastq_files:
        stats = load_fastq_stats(fq)
        total_reads += stats['n_reads']
        total_length += stats['mean_length'] * stats['n_reads']
        total_quality += stats['mean_quality'] * stats['n_reads'] * stats['mean_length']

    avg_length = total_length / total_reads if total_reads > 0 else 0
    avg_quality = total_quality / (total_reads * avg_length) if total_reads > 0 and avg_length > 0 else 0

    # Load ground truth if available for complexity estimation
    truth_file = Path(data_dir) / 'ground_truth.tsv'
    if truth_file.exists():
        truth_df = pd.read_csv(truth_file, sep='\t')
        complexity = estimate_complexity(truth_df)
    else:
        complexity = 0.0  # Unknown without ground truth

    return {
        'sequencing_depth': total_reads,
        'mean_read_length': avg_length,
        'mean_quality': avg_quality,
        'estimated_complexity': complexity,
    }
