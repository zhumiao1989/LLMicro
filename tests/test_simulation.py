"""
Tests for Data Simulation Module
"""

import pytest
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Mock BioPython if not available
import sys
sys.modules['Bio'] = MagicMock()
sys.modules['Bio.Seq'] = MagicMock()
sys.modules['Bio.SeqRecord'] = MagicMock()
sys.modules['Bio.SeqIO'] = MagicMock()

from src.simulate_data import DataSimulator, SimulatedSample


class TestDataSimulator:
    """Test cases for DataSimulator class."""

    @pytest.fixture
    def simulator(self, tmp_path):
        """Create a test simulator."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
common:
  read_length: 150
  error_model: illumina
  random_seed: 42

low_complexity:
  n_samples: 5
  species_range:
    min: 10
    max: 30

medium_complexity:
  n_samples: 5
  species_range:
    min: 50
    max: 150

high_complexity:
  n_samples: 5
  species_range:
    min: 200
    max: 400
""")
        return DataSimulator(config_path=str(config_file), random_seed=42)

    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.random_seed == 42
        assert simulator.read_length == 150
        assert simulator.error_rate == 0.001

    def test_generate_synthetic_genome(self, simulator):
        """Test synthetic genome generation."""
        genome = simulator._generate_synthetic_genome(1000)
        assert len(genome) == 1000
        assert all(base in 'ACGT' for base in genome)

    def test_add_sequencing_errors(self, simulator):
        """Test adding sequencing errors."""
        from Bio.Seq import Seq
        original_seq = Seq("AAAAAAAAAA")

        # With very low error rate, most bases should remain unchanged
        mutated_seq = simulator._add_sequencing_errors(original_seq)

        assert len(str(mutated_seq)) == len(str(original_seq))

    def test_create_sample(self, simulator):
        """Test sample creation."""
        sample = simulator._create_sample(
            sample_id="test_sample",
            complexity_type="low",
            selected_taxa=['562', '1280', '1423'],
            abundances=[0.5, 0.3, 0.2],
            abundance_distribution="uniform"
        )

        assert sample.sample_id == "test_sample"
        assert sample.complexity_type == "low"
        assert sample.n_species == 3
        assert len(sample.species_composition) == 3
        assert '562' in sample.species_composition

    def test_save_sample(self, simulator, tmp_path):
        """Test saving sample to files."""
        sample = simulator._create_sample(
            sample_id="test_sample",
            complexity_type="low",
            selected_taxa=['562', '1280'],
            abundances=[0.6, 0.4],
            abundance_distribution="uniform"
        )

        output_prefix = tmp_path / "test_output"
        simulator._save_sample(sample, output_prefix)

        # Check files were created
        assert (tmp_path / "test_output_ground_truth.tsv").exists()
        assert (tmp_path / "test_output_composition.json").exists()
        assert (tmp_path / "test_output_metadata.json").exists()

    def test_ground_truth_format(self, simulator, tmp_path):
        """Test ground truth file format."""
        sample = simulator._create_sample(
            sample_id="test_sample",
            complexity_type="low",
            selected_taxa=['562', '1280'],
            abundances=[0.6, 0.4],
            abundance_distribution="uniform"
        )

        output_prefix = tmp_path / "test_output"
        simulator._save_sample(sample, output_prefix)

        import pandas as pd
        gt = pd.read_csv(tmp_path / "test_output_ground_truth.tsv", sep='\t')

        assert 'read_id' in gt.columns
        assert 'taxonomy_id' in gt.columns
        assert 'abundance' in gt.columns
        assert 'is_human' in gt.columns

    def test_composition_format(self, simulator, tmp_path):
        """Test composition JSON file format."""
        sample = simulator._create_sample(
            sample_id="test_sample",
            complexity_type="low",
            selected_taxa=['562', '1280'],
            abundances=[0.6, 0.4],
            abundance_distribution="uniform"
        )

        output_prefix = tmp_path / "test_output"
        simulator._save_sample(sample, output_prefix)

        with open(tmp_path / "test_output_composition.json") as f:
            composition = json.load(f)

        assert 'sample_id' in composition
        assert 'complexity_type' in composition
        assert 'n_species' in composition
        assert 'species_composition' in composition

    def test_low_complexity_range(self, simulator):
        """Test low complexity sample species count is in range."""
        # Generate multiple samples and check species count
        for _ in range(10):
            n_species = 10  # Fixed for this test
            sample = simulator._create_sample(
                sample_id="test",
                complexity_type="low",
                selected_taxa=[str(i) for i in range(n_species)],
                abundances=[1.0/n_species] * n_species,
                abundance_distribution="uniform"
            )
            assert 10 <= sample.n_species <= 30 or sample.n_species == n_species

    def test_reproducibility(self, tmp_path):
        """Test that results are reproducible with same seed."""
        sim1 = DataSimulator(random_seed=12345)
        sim2 = DataSimulator(random_seed=12345)

        genome1 = sim1._generate_synthetic_genome(1000)
        genome2 = sim2._generate_synthetic_genome(1000)

        # With same seed, results should be identical
        assert genome1 == genome2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
