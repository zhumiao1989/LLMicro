#!/usr/bin/env python
"""
Centrifuge Runner for LLMicro

Runs Centrifuge classification with specified parameters.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import click


def run_centrifuge(
    input_reads: str,
    database: str,
    output_dir: str,
    k: int = 1,
    min_hitlen: int = 15,
    threads: int = 8,
    sample_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Centrifuge classification.

    Args:
        input_reads: Path to input FASTQ file
        database: Path to Centrifuge database (prefix)
        output_dir: Output directory
        k: Report up to k distinct alignments
        min_hitlen: Minimum hit length
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
        'centrifuge',
        '-x', str(db_path),
        '-p', str(threads),
        '-k', str(k),
        '--min-hitlen', str(min_hitlen),
        '--report-file', str(report_file),
        '-U', str(input_path)
    ]

    logger.info(f"Running Centrifuge: {' '.join(cmd)}")

    # Run and redirect stdout to classification file
    with open(classification_file, 'w') as out_f:
        result = subprocess.run(cmd, capture_output=out_f, text=True)

    if result.returncode != 0:
        logger.error(f"Centrifuge failed: {result.stderr}")
        raise RuntimeError(f"Centrifuge failed: {result.stderr}")

    # Parse stats from report
    stats = parse_centrifuge_report(report_file)
    stats['sample_id'] = sample_id
    stats['k'] = k
    stats['min_hitlen'] = min_hitlen

    logger.info(f"Centrifuge completed for {sample_id}")

    return stats


def parse_centrifuge_report(report_file: Path) -> Dict[str, Any]:
    """Parse Centrifuge report file."""
    stats = {'total_reads': 0, 'classified_reads': 0}

    try:
        with open(report_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    reads = int(parts[4])
                    stats['total_reads'] += reads
                    if parts[0] != 'unclassified':
                        stats['classified_reads'] += reads
    except Exception as e:
        logger.warning(f"Could not parse report: {e}")

    return stats


@click.command()
@click.option('--input', '-i', required=True, help='Input FASTQ file')
@click.option('--database', '-d', required=True, help='Centrifuge database prefix')
@click.option('--output', '-o', required=True, help='Output directory')
@click.option('--k', '-k', default=1, type=int, help='Max distinct alignments')
@click.option('--min-hitlen', '-m', default=15, type=int, help='Minimum hit length')
@click.option('--threads', '-t', default=8, type=int, help='Number of threads')
@click.option('--sample-id', '-s', default=None, help='Sample ID')
def main(input, database, output, k, min_hitlen, threads, sample_id):
    """Run Centrifuge classification."""
    stats = run_centrifuge(
        input_reads=input,
        database=database,
        output_dir=output,
        k=k,
        min_hitlen=min_hitlen,
        threads=threads,
        sample_id=sample_id
    )

    click.echo(f"Centrifuge completed:")
    click.echo(f"  Total reads: {stats.get('total_reads', 'N/A')}")
    click.echo(f"  Classified: {stats.get('classified_reads', 'N/A')}")


if __name__ == '__main__':
    main()
