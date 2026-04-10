#!/usr/bin/env python
"""
PathSeq Runner for LLMicro

Runs GATK PathSeq classification with specified parameters.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import click


def run_pathseq(
    input_reads: str,
    reference: str,
    host_reference: str,
    microbe_reference: str,
    output_dir: str,
    min_clipped_read_length: int = 30,
    host_min_identity: float = 95.0,
    min_score_identity: float = 0.95,
    identity_margin: float = 0.05,
    threads: int = 8,
    driver_memory: str = "4G",
    executor_memory: str = "8G",
    sample_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run GATK PathSeq classification.

    Args:
        input_reads: Path to input FASTQ file
        reference: Path to reference genome
        host_reference: Path to host reference
        microbe_reference: Path to microbe reference
        output_dir: Output directory
        min_clipped_read_length: Minimum clipped read length
        host_min_identity: Host minimum identity
        min_score_identity: Minimum score identity
        identity_margin: Identity margin
        threads: Number of threads
        driver_memory: Spark driver memory
        executor_memory: Spark executor memory
        sample_id: Sample identifier

    Returns:
        Dictionary with run statistics
    """
    input_path = Path(input_reads)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if sample_id is None:
        sample_id = input_path.stem

    # Output files
    output_prefix = output_path / sample_id

    # Build command
    cmd = [
        'gatk', 'PathSeqPipelineSpark',
        '--input', str(input_path),
        '--reference', reference,
        '--host-reference', host_reference,
        '--microbe-reference', microbe_reference,
        '--output', str(output_prefix),
        '--min-clipped-read-length', str(min_clipped_read_length),
        '--host-min-identity', str(host_min_identity),
        '--min-score-identity', str(min_score_identity),
        '--identity-margin', str(identity_margin),
        '--spark-runner', 'SPARK',
        '--conf', f'spark.driver.memory={driver_memory}',
        '--conf', f'spark.executor.memory={executor_memory}',
        '--',
        '--spark-master', f'local[{threads}]'
    ]

    logger.info(f"Running PathSeq: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"PathSeq failed: {result.stderr}")
        raise RuntimeError(f"PathSeq failed: {result.stderr}")

    # Parse stats from output
    stats = parse_pathseq_output(f"{output_prefix}_profile.tsv")
    stats['sample_id'] = sample_id
    stats['min_clipped_read_length'] = min_clipped_read_length
    stats['host_min_identity'] = host_min_identity
    stats['min_score_identity'] = min_score_identity
    stats['identity_margin'] = identity_margin

    logger.info(f"PathSeq completed for {sample_id}")

    return stats


def parse_pathseq_output(profile_file: str) -> Dict[str, Any]:
    """Parse PathSeq profile output."""
    stats = {'total_reads': 0, 'classified_reads': 0}

    try:
        import pandas as pd
        df = pd.read_csv(profile_file, sep='\t')

        stats['total_reads'] = df['count'].sum() if 'count' in df.columns else 0
        stats['classified_reads'] = stats['total_reads']

        # Get top taxa
        if 'taxon' in df.columns:
            stats['top_taxon'] = df.iloc[0]['taxon'] if len(df) > 0 else None

    except Exception as e:
        logger.warning(f"Could not parse profile: {e}")

    return stats


@click.command()
@click.option('--input', '-i', required=True, help='Input FASTQ file')
@click.option('--reference', '-r', required=True, help='Reference genome')
@click.option('--host-reference', '-H', required=True, help='Host reference')
@click.option('--microbe-reference', '-m', required=True, help='Microbe reference')
@click.option('--output', '-o', required=True, help='Output directory')
@click.option('--min-clipped-read-length', default=30, type=int, help='Min clipped read length')
@click.option('--host-min-identity', default=95.0, type=float, help='Host min identity')
@click.option('--min-score-identity', default=0.95, type=float, help='Min score identity')
@click.option('--identity-margin', default=0.05, type=float, help='Identity margin')
@click.option('--threads', '-t', default=8, type=int, help='Number of threads')
@click.option('--sample-id', '-s', default=None, help='Sample ID')
def main(input, reference, host_reference, microbe_reference, output,
         min_clipped_read_length, host_min_identity, min_score_identity,
         identity_margin, threads, sample_id):
    """Run PathSeq classification."""
    stats = run_pathseq(
        input_reads=input,
        reference=reference,
        host_reference=host_reference,
        microbe_reference=microbe_reference,
        output_dir=output,
        min_clipped_read_length=min_clipped_read_length,
        host_min_identity=host_min_identity,
        min_score_identity=min_score_identity,
        identity_margin=identity_margin,
        threads=threads,
        sample_id=sample_id
    )

    click.echo(f"PathSeq completed:")
    click.echo(f"  Total reads: {stats.get('total_reads', 'N/A')}")
    click.echo(f"  Classified: {stats.get('classified_reads', 'N/A')}")


if __name__ == '__main__':
    main()
