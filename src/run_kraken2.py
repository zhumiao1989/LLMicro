#!/usr/bin/env python
"""
Kraken2 Runner for LLMicro

Runs Kraken2 classification with specified parameters.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import click


def run_kraken2(
    input_reads: str,
    database: str,
    output_dir: str,
    confidence: float = 0.0,
    minimum_hit_groups: int = 1,
    threads: int = 8,
    sample_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Kraken2 classification.

    Args:
        input_reads: Path to input FASTQ file
        database: Path to Kraken2 database
        output_dir: Output directory
        confidence: Confidence threshold (0.0-1.0)
        minimum_hit_groups: Minimum hit groups (1-5)
        threads: Number of threads
        sample_id: Sample identifier

    Returns:
        Dictionary with run statistics
    """
    input_path = Path(input_reads)
    db_path = Path(database)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if sample_id is None:
        sample_id = input_path.stem

    # Output files
    report_file = output_path / f"{sample_id}_report.tsv"
    classification_file = output_path / f"{sample_id}_classification.tsv"

    # Build command
    cmd = [
        'kraken2',
        '--db', str(db_path),
        '--threads', str(threads),
        '--report', str(report_file),
        '--output', str(classification_file),
        '--confidence', str(confidence),
        '--minimum-hit-groups', str(minimum_hit_groups),
        str(input_path)
    ]

    logger.info(f"Running Kraken2: {' '.join(cmd)}")

    # Run
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Kraken2 failed: {result.stderr}")
        raise RuntimeError(f"Kraken2 failed: {result.stderr}")

    # Parse stats from stderr
    stats = parse_kraken2_stats(result.stderr)
    stats['sample_id'] = sample_id
    stats['confidence'] = confidence
    stats['minimum_hit_groups'] = minimum_hit_groups

    logger.info(f"Kraken2 completed for {sample_id}")

    return stats


def parse_kraken2_stats(stderr: str) -> Dict[str, Any]:
    """Parse Kraken2 statistics from stderr output."""
    stats = {}

    for line in stderr.split('\n'):
        if 'reads classified' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                stats['reads_classified'] = int(parts[1].strip().split()[0].replace(',', ''))
        elif 'reads unclassified' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                stats['reads_unclassified'] = int(parts[1].strip().split()[0].replace(',', ''))

    return stats


@click.command()
@click.option('--input', '-i', required=True, help='Input FASTQ file')
@click.option('--database', '-d', required=True, help='Kraken2 database path')
@click.option('--output', '-o', required=True, help='Output directory')
@click.option('--confidence', '-c', default=0.0, type=float, help='Confidence threshold')
@click.option('--min-hit-groups', '-g', default=1, type=int, help='Minimum hit groups')
@click.option('--threads', '-t', default=8, type=int, help='Number of threads')
@click.option('--sample-id', '-s', default=None, help='Sample ID')
def main(input, database, output, confidence, min_hit_groups, threads, sample_id):
    """Run Kraken2 classification."""
    stats = run_kraken2(
        input_reads=input,
        database=database,
        output_dir=output,
        confidence=confidence,
        minimum_hit_groups=min_hit_groups,
        threads=threads,
        sample_id=sample_id
    )

    click.echo(f"Kraken2 completed:")
    click.echo(f"  Classified: {stats.get('reads_classified', 'N/A')}")
    click.echo(f"  Unclassified: {stats.get('reads_unclassified', 'N/A')}")


if __name__ == '__main__':
    main()
