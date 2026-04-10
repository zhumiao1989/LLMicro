"""
Tests for LLM Recommender Module
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Mock anthropic and openai before importing llm_recommender
import sys
sys.modules['anthropic'] = MagicMock()
sys.modules['openai'] = MagicMock()

from src.llm_recommender import LLMRecommender, ParameterRecommendation


class TestLLMRecommender:
    """Test cases for LLMRecommender class."""

    @pytest.fixture
    def sample_features(self):
        """Sample features for testing."""
        return {
            'sequencing_depth': 1000000,
            'mean_read_length': 150,
            'mean_quality': 35,
            'estimated_complexity': 3.5
        }

    @pytest.fixture
    def recommender(self, tmp_path):
        """Create a test recommender."""
        # Create a minimal config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
parameter_ranges:
  kraken2:
    confidence:
      min: 0.0
      max: 1.0
      default: 0.0
    minimum_hit_groups:
      min: 1
      max: 5
      default: 1
  centrifuge:
    k:
      min: 1
      max: 10
      default: 1
    min_hitlen:
      min: 15
      max: 35
      default: 15
  pathseq:
    min_clipped_read_length:
      min: 30
      max: 40
      default: 30
    host_min_identity:
      min: 80
      max: 100
      default: 95
    min_score_identity:
      min: 0.80
      max: 1.0
      default: 0.95
    identity_margin:
      min: 0.0
      max: 0.1
      default: 0.05
""")
        return LLMRecommender(config_path=str(config_file), provider='anthropic')

    def test_initialization(self, recommender):
        """Test recommender initialization."""
        assert recommender.provider == 'anthropic'
        assert recommender.parameter_ranges is not None
        assert 'kraken2' in recommender.parameter_ranges
        assert 'centrifuge' in recommender.parameter_ranges
        assert 'pathseq' in recommender.parameter_ranges

    def test_get_default_parameters_kraken2(self, recommender):
        """Test getting default parameters for Kraken2."""
        defaults = recommender.get_default_parameters('kraken2')
        assert defaults['confidence'] == 0.0
        assert defaults['minimum_hit_groups'] == 1

    def test_get_default_parameters_centrifuge(self, recommender):
        """Test getting default parameters for Centrifuge."""
        defaults = recommender.get_default_parameters('centrifuge')
        assert defaults['k'] == 1
        assert defaults['min_hitlen'] == 15

    def test_get_default_parameters_pathseq(self, recommender):
        """Test getting default parameters for PathSeq."""
        defaults = recommender.get_default_parameters('pathseq')
        assert defaults['min_clipped_read_length'] == 30
        assert defaults['host_min_identity'] == 95
        assert defaults['min_score_identity'] == 0.95
        assert defaults['identity_margin'] == 0.05

    def test_invalid_tool(self, recommender):
        """Test handling of invalid tool name."""
        with pytest.raises(ValueError) as excinfo:
            recommender.get_default_parameters('invalid_tool')
        assert 'Unknown tool' in str(excinfo.value)

    @patch.object(LLMRecommender, '_call_llm')
    def test_recommend_kraken2(self, mock_call_llm, recommender, sample_features):
        """Test Kraken2 parameter recommendation."""
        # Mock LLM response
        mock_response = """
```json
{
    "recommended_parameters": {
        "confidence": 0.1,
        "minimum_hit_groups": 2
    },
    "reasoning": "Low complexity sample allows for more stringent filtering",
    "expected_tradeoffs": {
        "precision": "increase",
        "recall": "stable",
        "speed": "similar",
        "memory": "similar"
    }
}
```
"""
        mock_call_llm.return_value = mock_response

        rec = recommender.recommend('kraken2', sample_features)

        assert rec.tool == 'kraken2'
        assert rec.parameters['confidence'] == 0.1
        assert rec.parameters['minimum_hit_groups'] == 2
        assert 'Low complexity' in rec.reasoning

    @patch.object(LLMRecommender, '_call_llm')
    def test_recommend_centrifuge(self, mock_call_llm, recommender, sample_features):
        """Test Centrifuge parameter recommendation."""
        mock_response = """
```json
{
    "recommended_parameters": {
        "k": 3,
        "min_hitlen": 20
    },
    "reasoning": "Medium complexity sample requires balanced parameters",
    "expected_tradeoffs": {
        "precision": "increase",
        "recall": "stable"
    }
}
```
"""
        mock_call_llm.return_value = mock_response

        rec = recommender.recommend('centrifuge', sample_features)

        assert rec.tool == 'centrifuge'
        assert rec.parameters['k'] == 3
        assert rec.parameters['min_hitlen'] == 20

    @patch.object(LLMRecommender, '_call_llm')
    def test_recommend_pathseq(self, mock_call_llm, recommender, sample_features):
        """Test PathSeq parameter recommendation."""
        mock_response = """
```json
{
    "recommended_parameters": {
        "min_clipped_read_length": 35,
        "host_min_identity": 98,
        "min_score_identity": 0.98,
        "identity_margin": 0.02
    },
    "reasoning": "High host background requires stringent host filtering",
    "expected_tradeoffs": {
        "precision": "increase",
        "recall": "decrease"
    }
}
```
"""
        mock_call_llm.return_value = mock_response

        rec = recommender.recommend('pathseq', sample_features)

        assert rec.tool == 'pathseq'
        assert rec.parameters['min_clipped_read_length'] == 35
        assert rec.parameters['host_min_identity'] == 98

    def test_save_recommendations(self, recommender, tmp_path):
        """Test saving recommendations to file."""
        recommendations = [
            ParameterRecommendation(
                tool='kraken2',
                parameters={'confidence': 0.1, 'minimum_hit_groups': 2},
                reasoning='Test reasoning',
                expected_tradeoffs={'precision': 'increase'},
                confidence=0.8
            )
        ]

        output_file = tmp_path / "test_recommendations.json"
        recommender.save_recommendations(recommendations, str(output_file))

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]['tool'] == 'kraken2'
        assert data[0]['parameters']['confidence'] == 0.1

    def test_parse_malformed_json(self, recommender):
        """Test handling of malformed JSON response."""
        malformed_response = "This is not valid JSON"

        rec = recommender._parse_response(malformed_response, 'kraken2')

        # Should fall back to defaults
        assert rec.parameters['confidence'] == 0.0
        assert 'Failed to parse' in rec.reasoning
        assert rec.confidence == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
