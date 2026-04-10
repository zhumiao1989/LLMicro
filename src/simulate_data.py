"""
Simulated Data Generator for LLMicro Benchmarking

Generates three types of simulated metagenomic datasets:
1. Low complexity (10-30 species, uniform abundance)
2. Medium complexity (50-150 species, lognormal abundance)
3. High complexity (200-400 species, stepped abundance)
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

# Try to import BioPython for sequence handling
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logger.warning("BioPython not installed. Sequence generation will be limited.")

from .utils.io import load_config


@dataclass
class SimulatedSample:
    """Data class for a simulated metagenomic sample."""
    sample_id: str
    complexity_type: str  # low, medium, high
    n_species: int
    abundance_distribution: str
    species_composition: Dict[str, float]  # taxonomy_id -> abundance
    ground_truth: pd.DataFrame = field(default_factory=pd.DataFrame)
    reads: List[SeqRecord] = field(default_factory=list)


class DataSimulator:
    """
    Simulated metagenomic data generator.

    Generates realistic metagenomic reads with known ground truth
    for benchmarking classification tools.
    """

    # Example genome sequences (shortened for simulation)
    # In practice, these would come from reference databases
    EXAMPLE_GENOMES = {
        # Bacteria
        '562': 'ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG',  # E. coli
        '1280': 'ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC',  # S. aureus
        '1423': 'ATGGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGC',  # B. subtilis
        '287': 'ATGPSEUDOMONASAERUGINOSAGENOMESEQUENCEPSEUDOMONASAERUGINOSAGENOMESEQ',  # P. aeruginosa
        '590': 'ATGSALMONELLAENTERICAGENOMESEQUENCESALMONELLAENTERICAGENOMESEQUENCE',  # S. enterica
        '1585': 'ATGLACTOBACILLUSFERMENTUMGENOMESENUENCELACTOBACILLUSFERMENTUMGENOM',  # L. fermentum
        '1351': 'ATGENTEROCOCCUSFAECALISGENOMESEQUENCEENTEROCOCCUSFAECALISGENOMESEQ',  # E. faecalis
        '1639': 'ATGLISTERIAMONOCYTOGENESGENOMESENUENCELISTERIAMONOCYTOGENESGENOME',  # L. monocytogenes
        # Fungi
        '4932': 'ATGSACCHAROMYCESCEREVISIAEYEASTGENOMESEQUENCESACCHAROMYCESCEREVISIA',  # S. cerevisiae
        '5207': 'ATGCRYPTOCOCCUSNEOFORMANSGENOMESEQUENCECRYPTOCOCCUSNEOFORMANSGENOM',  # C. neoformans
        # Phage
        '10847': 'ATGPHIX174BACTERIOPHAGEGENOMESEQUENCEPHIX174BACTERIOPHAGEGENOMESEQ',  # PhiX174
        '10710': 'ATGLAMBDAPHAGEGENOMESEQUENCELAMBDAPHAGEGENOMESENUENCELAMBDAPHAGEG',  # Lambda phage
    }

    # Human genome placeholder (for contamination)
    HUMAN_GENOME_ID = 'hg38'

    def __init__(
        self,
        config_path: Optional[str] = None,
        random_seed: int = 42,
        read_length: int = 150,
        error_rate: float = 0.001
    ):
        """
        Initialize the data simulator.

        Args:
            config_path: Path to simulation configuration file
            random_seed: Random seed for reproducibility
            read_length: Length of simulated reads
            error_rate: Sequencing error rate
        """
        self.config = {}
        if config_path:
            self.config = load_config(config_path)

        self.random_seed = random_seed
        self.read_length = read_length
        self.error_rate = error_rate

        # Set random seeds
        np.random.seed(random_seed)
        random.seed(random_seed)

        logger.info(f"Initialized DataSimulator with seed={random_seed}, read_length={read_length}")

    def generate_low_complexity_samples(
        self,
        n_samples: int = 20,
        output_dir: str = 'data/simulated/low_complexity'
    ) -> List[SimulatedSample]:
        """
        Generate low complexity samples.

        Args:
            n_samples: Number of samples to generate
            output_dir: Output directory for generated files

        Returns:
            List of SimulatedSample objects
        """
        logger.info(f"Generating {n_samples} low complexity samples")

        samples = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            # Random number of species (10-30)
            n_species = random.randint(10, 30)

            # Select random species from available genomes
            available_taxa = list(self.EXAMPLE_GENOMES.keys())
            selected_taxa = random.sample(available_taxa, min(n_species, len(available_taxa)))

            # If we need more species than available, generate synthetic ones
            while len(selected_taxa) < n_species:
                new_taxon = f"synthetic_{len(selected_taxa)}"
                selected_taxa.append(new_taxon)
                self.EXAMPLE_GENOMES[new_taxon] = self._generate_synthetic_genome(1000)

            # Uniform abundance distribution
            abundances = np.ones(len(selected_taxa)) / len(selected_taxa)

            sample = self._create_sample(
                sample_id=f"low_complexity_{i+1:03d}",
                complexity_type="low",
                selected_taxa=selected_taxa,
                abundances=abundances,
                abundance_distribution="uniform"
            )

            samples.append(sample)

            # Save to files
            self._save_sample(sample, output_path / f"sample_{i+1:03d}")

        logger.info(f"Generated {n_samples} low complexity samples to {output_dir}")
        return samples

    def generate_medium_complexity_samples(
        self,
        n_samples: int = 20,
        output_dir: str = 'data/simulated/medium_complexity'
    ) -> List[SimulatedSample]:
        """
        Generate medium complexity samples with lognormal abundance distribution.

        Args:
            n_samples: Number of samples to generate
            output_dir: Output directory

        Returns:
            List of SimulatedSample objects
        """
        logger.info(f"Generating {n_samples} medium complexity samples")

        samples = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            # Random number of species (50-150)
            n_species = random.randint(50, 150)

            # Select species
            available_taxa = list(self.EXAMPLE_GENOMES.keys())
            selected_taxa = random.sample(available_taxa, min(n_species, len(available_taxa)))

            # Add synthetic taxa if needed
            while len(selected_taxa) < n_species:
                new_taxon = f"synthetic_{len(selected_taxa)}"
                selected_taxa.append(new_taxon)
                self.EXAMPLE_GENOMES[new_taxon] = self._generate_synthetic_genome(1000)

            # Lognormal abundance distribution (typical microbiome)
            raw_abundances = np.random.lognormal(mean=0, sigma=2, size=n_species)
            abundances = raw_abundances / raw_abundances.sum()

            sample = self._create_sample(
                sample_id=f"medium_complexity_{i+1:03d}",
                complexity_type="medium",
                selected_taxa=selected_taxa,
                abundances=abundances,
                abundance_distribution="lognormal"
            )

            samples.append(sample)
            self._save_sample(sample, output_path / f"sample_{i+1:03d}")

        logger.info(f"Generated {n_samples} medium complexity samples to {output_dir}")
        return samples

    def generate_high_complexity_samples(
        self,
        n_samples: int = 20,
        output_dir: str = 'data/simulated/high_complexity'
    ) -> List[SimulatedSample]:
        """
        Generate high complexity samples with stepped abundance distribution.

        Args:
            n_samples: Number of samples to generate
            output_dir: Output directory

        Returns:
            List of SimulatedSample objects
        """
        logger.info(f"Generating {n_samples} high complexity samples")

        samples = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            # Random number of species (200-400)
            n_species = random.randint(200, 400)

            # Select species
            available_taxa = list(self.EXAMPLE_GENOMES.keys())
            selected_taxa = random.sample(available_taxa, min(n_species, len(available_taxa)))

            # Add synthetic taxa if needed
            while len(selected_taxa) < n_species:
                new_taxon = f"synthetic_{len(selected_taxa)}"
                selected_taxa.append(new_taxon)
                self.EXAMPLE_GENOMES[new_taxon] = self._generate_synthetic_genome(1000)

            # Stepped abundance distribution (long-tail rare biosphere)
            # Divide species into abundance tiers
            n_tiers = 5
            tier_sizes = [int(n_species * 0.05), int(n_species * 0.1), int(n_species * 0.15),
                         int(n_species * 0.2), n_species - sum([int(n_species * 0.05), int(n_species * 0.1),
                                                                int(n_species * 0.15), int(n_species * 0.2)])]

            tier_abundances = [0.4, 0.25, 0.15, 0.1, 0.1]  # Relative weight of each tier

            abundances = []
            idx = 0
            for tier_idx, (tier_size, tier_weight) in enumerate(zip(tier_sizes, tier_abundances)):
                # Within each tier, uniform distribution
                tier_abund = np.ones(tier_size) * (tier_weight / tier_size) if tier_size > 0 else np.array([])
                abundances.extend(tier_abund.tolist())

            abundances = np.array(abundances[:n_species])
            abundances = abundances / abundances.sum()  # Normalize

            sample = self._create_sample(
                sample_id=f"high_complexity_{i+1:03d}",
                complexity_type="high",
                selected_taxa=selected_taxa,
                abundances=abundances,
                abundance_distribution="stepped"
            )

            samples.append(sample)
            self._save_sample(sample, output_path / f"sample_{i+1:03d}")

        logger.info(f"Generated {n_samples} high complexity samples to {output_dir}")
        return samples

    def generate_mock_community_samples(
        self,
        n_replicates: int = 10,
        output_dir: str = 'data/mock'
    ) -> List[SimulatedSample]:
        """
        Generate mock community samples with known composition.

        Args:
            n_replicates: Number of technical replicates
            output_dir: Output directory

        Returns:
            List of SimulatedSample objects
        """
        logger.info(f"Generating {n_replicates} mock community samples")

        samples = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Defined mock community composition (from config)
        mock_config = self.config.get('mock_community', {})
        composition = mock_config.get('composition', [])

        if not composition:
            # Use default composition
            composition = [
                {'name': 'Escherichia coli', 'taxonomy_id': '562', 'abundance': 0.15},
                {'name': 'Pseudomonas aeruginosa', 'taxonomy_id': '287', 'abundance': 0.10},
                {'name': 'Salmonella enterica', 'taxonomy_id': '590', 'abundance': 0.10},
                {'name': 'Staphylococcus aureus', 'taxonomy_id': '1280', 'abundance': 0.12},
                {'name': 'Bacillus subtilis', 'taxonomy_id': '1423', 'abundance': 0.10},
                {'name': 'Limosilactobacillus fermentum', 'taxonomy_id': '1585', 'abundance': 0.08},
                {'name': 'Enterococcus faecalis', 'taxonomy_id': '1351', 'abundance': 0.08},
                {'name': 'Listeria monocytogenes', 'taxonomy_id': '1639', 'abundance': 0.07},
                {'name': 'Saccharomyces cerevisiae', 'taxonomy_id': '4932', 'abundance': 0.10},
                {'name': 'Cryptococcus neoformans', 'taxonomy_id': '5207', 'abundance': 0.05},
                {'name': 'PhiX174', 'taxonomy_id': '10847', 'abundance': 0.03},
                {'name': 'Lambda phage', 'taxonomy_id': '10710', 'abundance': 0.02},
            ]

        selected_taxa = [str(c['taxonomy_id']) for c in composition]
        abundances = np.array([c['abundance'] for c in composition])

        for i in range(n_replicates):
            # Add small variation between replicates
            noise = np.random.normal(0, 0.02, size=len(abundances))
            noisy_abundances = np.clip(abundances + noise, 0.01, None)
            noisy_abundances = noisy_abundances / noisy_abundances.sum()

            sample = self._create_sample(
                sample_id=f"mock_replicate_{i+1:03d}",
                complexity_type="mock",
                selected_taxa=selected_taxa,
                abundances=noisy_abundances,
                abundance_distribution="defined"
            )

            samples.append(sample)
            self._save_sample(sample, output_path / f"replicate_{i+1:03d}")

        logger.info(f"Generated {n_replicates} mock community samples to {output_dir}")
        return samples

    def _create_sample(
        self,
        sample_id: str,
        complexity_type: str,
        selected_taxa: List[str],
        abundances: np.ndarray,
        abundance_distribution: str
    ) -> SimulatedSample:
        """Create a simulated sample with ground truth."""

        # Create species composition dictionary
        species_composition = {taxa: float(abund) for taxa, abund in zip(selected_taxa, abundances)}

        # Generate ground truth DataFrame
        n_total_reads = 100000  # Target depth per sample
        read_counts = (abundances * n_total_reads).astype(int)

        # Add 90% human contamination
        n_human_reads = int(n_total_reads * 0.9 / 0.1)  # 90% human, 10% microbial
        human_read_counts = np.array([n_human_reads])

        ground_truth_data = []

        # Microbial reads
        read_id = 0
        for taxa, count in zip(selected_taxa, read_counts):
            for _ in range(count):
                ground_truth_data.append({
                    'read_id': f"read_{read_id:08d}",
                    'taxonomy_id': taxa,
                    'abundance': species_composition[taxa],
                    'is_human': False
                })
                read_id += 1

        # Human reads
        for _ in range(n_human_reads):
            ground_truth_data.append({
                'read_id': f"read_{read_id:08d}",
                'taxonomy_id': self.HUMAN_GENOME_ID,
                'abundance': 0.9,
                'is_human': True
            })
            read_id += 1

        ground_truth = pd.DataFrame(ground_truth_data)

        # Generate simulated reads (simplified - in practice would use ART or similar)
        reads = self._generate_reads(selected_taxa, abundances, n_total_reads)

        return SimulatedSample(
            sample_id=sample_id,
            complexity_type=complexity_type,
            n_species=len(selected_taxa),
            abundance_distribution=abundance_distribution,
            species_composition=species_composition,
            ground_truth=ground_truth,
            reads=reads
        )

    def _generate_reads(
        self,
        taxa: List[str],
        abundances: np.ndarray,
        n_reads: int
    ) -> List[SeqRecord]:
        """Generate simulated reads from given taxa."""
        reads = []

        if not BIOPYTHON_AVAILABLE:
            # Return placeholder if BioPython not available
            logger.warning("BioPython not available, generating placeholder reads")
            return reads

        n_taxa_reads = (abundances * n_reads * 0.1).astype(int)  # 10% microbial

        for taxon_id, n_taxon_reads in zip(taxa, n_taxa_reads):
            genome_seq = self.EXAMPLE_GENOMES.get(taxon_id, '')
            if not genome_seq:
                continue

            genome = Seq(genome_seq)

            for i in range(n_taxon_reads):
                # Random start position
                if len(genome) > self.read_length:
                    start = random.randint(0, len(genome) - self.read_length)
                    read_seq = genome[start:start + self.read_length]
                else:
                    read_seq = genome

                # Add sequencing errors
                read_seq = self._add_sequencing_errors(read_seq)

                reads.append(SeqRecord(
                    read_seq,
                    id=f"read_{len(reads):08d}",
                    description=f"taxon={taxon_id}"
                ))

        return reads

    def _add_sequencing_errors(self, seq: Seq) -> Seq:
        """Add sequencing errors to a read."""
        seq_list = list(str(seq))

        for i in range(len(seq_list)):
            if random.random() < self.error_rate:
                # Substitution error
                original = seq_list[i]
                bases = ['A', 'C', 'G', 'T']
                new_base = random.choice([b for b in bases if b != original])
                seq_list[i] = new_base

        return Seq(''.join(seq_list))

    def _generate_synthetic_genome(self, length: int = 1000) -> str:
        """Generate a synthetic genome sequence."""
        bases = ['A', 'C', 'G', 'T']
        return ''.join(random.choice(bases) for _ in range(length))

    def _save_sample(self, sample: SimulatedSample, output_prefix: Path) -> None:
        """Save sample to files."""
        # Save ground truth
        ground_truth_path = f"{output_prefix}_ground_truth.tsv"
        sample.ground_truth.to_csv(ground_truth_path, sep='\t', index=False)
        logger.debug(f"Saved ground truth to {ground_truth_path}")

        # Save species composition as JSON
        composition_path = f"{output_prefix}_composition.json"
        with open(composition_path, 'w') as f:
            json.dump({
                'sample_id': sample.sample_id,
                'complexity_type': sample.complexity_type,
                'n_species': sample.n_species,
                'abundance_distribution': sample.abundance_distribution,
                'species_composition': sample.species_composition
            }, f, indent=2)
        logger.debug(f"Saved composition to {composition_path}")

        # Save reads as FASTQ if BioPython available
        if BIOPYTHON_AVAILABLE and sample.reads:
            reads_path = f"{output_prefix}_reads.fastq"
            SeqIO.write(sample.reads, reads_path, "fastq")
            logger.debug(f"Saved reads to {reads_path}")

        # Save sample metadata
        metadata_path = f"{output_prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'sample_id': sample.sample_id,
                'complexity_type': sample.complexity_type,
                'n_species': sample.n_species,
                'abundance_distribution': sample.abundance_distribution,
                'total_reads': len(sample.ground_truth),
                'microbial_reads': len(sample.reads)
            }, f, indent=2)
        logger.debug(f"Saved metadata to {metadata_path}")


def main():
    """CLI entry point."""
    import click

    @click.command()
    @click.option('--config', '-c', default='config/simulation_config.yaml', help='Config file')
    @click.option('--output', '-o', default='data/simulated', help='Output directory')
    @click.option('--type', '-t', 'sample_type', type=click.Choice(['low', 'medium', 'high', 'mock', 'all']), default='all')
    @click.option('--n-samples', '-n', default=20, help='Number of samples per type')
    @click.option('--seed', '-s', default=42, help='Random seed')
    def generate_data(config, output, sample_type, n_samples, seed):
        """Generate simulated metagenomic datasets."""
        simulator = DataSimulator(config_path=config, random_seed=seed)

        if sample_type in ['low', 'all']:
            simulator.generate_low_complexity_samples(n_samples, f"{output}/low_complexity")

        if sample_type in ['medium', 'all']:
            simulator.generate_medium_complexity_samples(n_samples, f"{output}/medium_complexity")

        if sample_type in ['high', 'all']:
            simulator.generate_high_complexity_samples(n_samples, f"{output}/high_complexity")

        if sample_type in ['mock', 'all']:
            simulator.generate_mock_community_samples(10, f"{output}/mock")

        click.echo(f"Data generation complete. Output: {output}")

    generate_data()


if __name__ == '__main__':
    main()
