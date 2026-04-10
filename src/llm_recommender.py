"""
LLM-based Parameter Recommender for Metagenomic Classification Tools

LLMicro: A context-aware parameter recommendation framework for robust metagenomic profiling
that combines large language models (LLMs) with retrieval-augmented generation (RAG).
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

from .utils.io import load_config, get_sample_features
from .utils.prompts import get_parameter_recommendation_prompt
from .rag_knowledge_base import RAGKnowledgeBase


@dataclass
class ParameterRecommendation:
    """Data class for parameter recommendation results."""
    tool: str
    parameters: Dict[str, float]
    reasoning: str
    expected_tradeoffs: Dict[str, str]
    confidence: float = 0.0


class LLMRecommender:
    """
    LLMicro: A context-aware parameter recommendation framework for robust metagenomic profiling.

    LLMicro combines large language models (LLMs) with retrieval-augmented generation (RAG)
    to generate tool-specific parameter configurations adapted to the current metagenomic context.

    Components:
    - Parameter knowledge acquisition and RAG-based evidence construction
    - Sample-aware feature extraction and parameter search space definition
    - Evidence-constrained LLM recommendation with hallucination control
    - Resource-aware optimization strategy

    Supports:
    - Kraken2
    - Centrifuge
    - PathSeq

    LLM Providers:
    - Anthropic (Claude)
    - OpenAI (GPT)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        use_rag: bool = True,
        knowledge_dir: Optional[str] = None
    ):
        """
        Initialize the LLMicro recommender.

        Args:
            config_path: Path to YAML configuration file
            provider: LLM provider ('anthropic' or 'openai')
            model: Model name to use
            api_key: API key (if not set in environment)
            use_rag: Whether to use RAG-based evidence retrieval
            knowledge_dir: Directory for RAG knowledge base (default: data/knowledge)
        """
        self.config = {}
        if config_path:
            self.config = load_config(config_path)
            logger.info(f"Loaded configuration from {config_path}")

        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key(provider)
        self.use_rag = use_rag

        # Initialize client
        self.client = self._init_client()

        # Initialize RAG knowledge base
        if use_rag:
            kb_dir = knowledge_dir or self.config.get('knowledge_dir', 'data/knowledge')
            index_dir = self.config.get('index_dir', 'data/index')
            self.knowledge_base = RAGKnowledgeBase(
                documents_dir=kb_dir,
                index_dir=index_dir
            )
            logger.info(f"Initialized RAG knowledge base with {len(self.knowledge_base.chunks)} chunks")
        else:
            self.knowledge_base = None
            logger.info("RAG disabled. Using LLM internal knowledge only.")

        # Default parameter ranges
        self.parameter_ranges = self.config.get('parameter_ranges', {
            'kraken2': {
                'confidence': {'min': 0.0, 'max': 1.0, 'default': 0.0},
                'minimum_hit_groups': {'min': 1, 'max': 5, 'default': 1}
            },
            'centrifuge': {
                'k': {'min': 1, 'max': 10, 'default': 1},
                'min_hitlen': {'min': 15, 'max': 35, 'default': 15}
            },
            'pathseq': {
                'min_clipped_read_length': {'min': 30, 'max': 40, 'default': 30},
                'host_min_identity': {'min': 80, 'max': 100, 'default': 95},
                'min_score_identity': {'min': 0.80, 'max': 1.0, 'default': 0.95},
                'identity_margin': {'min': 0.0, 'max': 0.1, 'default': 0.05}
            }
        })

        logger.info(f"Initialized LLMicro with {provider}/{model}, RAG={use_rag}")

    def _get_api_key(self, provider: str) -> str:
        """Get API key from environment or config."""
        if provider == 'anthropic':
            key = os.environ.get('ANTHROPIC_API_KEY', '')
            if not key:
                logger.warning("ANTHROPIC_API_KEY not set in environment")
            return key
        elif provider == 'openai':
            key = os.environ.get('OPENAI_API_KEY', '')
            if not key:
                logger.warning("OPENAI_API_KEY not set in environment")
            return key
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.provider == 'anthropic':
            if anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            return anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == 'openai':
            if openai is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            return openai.OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def recommend(
        self,
        tool: str,
        sample_features: Dict[str, Any],
        database_info: Optional[Dict[str, Any]] = None,
        resource_constraints: Optional[Dict[str, Any]] = None
    ) -> ParameterRecommendation:
        """
        Get parameter recommendation for a specific tool.

        Args:
            tool: Tool name ('kraken2', 'centrifuge', 'pathseq')
            sample_features: Sample characteristics
            database_info: Database information
            resource_constraints: Resource constraints

        Returns:
            ParameterRecommendation object
        """
        if tool not in self.parameter_ranges:
            raise ValueError(f"Unknown tool: {tool}. Supported: {list(self.parameter_ranges.keys())}")

        # Use defaults if not provided
        if database_info is None:
            database_info = self.config.get('database', {
                'source': 'GenBank 202404',
                'taxa': ['Bacteria', 'Fungi', 'Parasites', 'Viruses']
            })

        if resource_constraints is None:
            resource_constraints = self.config.get('resource_constraints', {
                'max_memory_gb': 64,
                'max_threads': 16,
                'target_speed_m_reads_per_min': 1.0
            })

        # Retrieve evidence using RAG (if enabled)
        retrieved_evidence = {}
        if self.use_rag and self.knowledge_base:
            logger.info(f"Retrieving evidence for {tool}...")
            parameter_names = list(self.parameter_ranges.get(tool, {}).keys())
            retrieved_evidence = self.knowledge_base.retrieve_multi_round(
                tool_name=tool,
                parameter_names=parameter_names,
                top_k=3
            )
            logger.info(f"Retrieved evidence for {len(retrieved_evidence)} parameters")

        # Generate prompt with retrieved evidence
        prompt = get_parameter_recommendation_prompt(
            tool=tool,
            sample_features=sample_features,
            database_info=database_info,
            resource_constraints=resource_constraints,
            parameter_ranges=self.parameter_ranges,
            retrieved_evidence=retrieved_evidence
        )

        # Call LLM API
        response = self._call_llm(prompt)

        # Parse response
        recommendation = self._parse_response(response, tool)

        logger.info(f"Generated recommendation for {tool}: {recommendation.parameters}")

        return recommendation

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API and get response."""
        if self.provider == 'anthropic':
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert bioinformatics assistant specializing in metagenomic classification parameter optimization."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.content[0].text

        elif self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=2048,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert bioinformatics assistant specializing in metagenomic classification parameter optimization."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response: str, tool: str) -> ParameterRecommendation:
        """Parse LLM response into structured recommendation."""
        # Try to extract JSON from response
        import re

        # Look for JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code block
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        try:
            data = json.loads(json_str)

            params = data.get('recommended_parameters', {})
            reasoning = data.get('reasoning', 'No reasoning provided')
            tradeoffs = data.get('expected_tradeoffs', {})
            confidence = data.get('confidence', 0.5)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return default parameters
            params = {}
            for param, info in self.parameter_ranges.get(tool, {}).items():
                params[param] = info.get('default', 0)
            reasoning = "Failed to parse LLM response, using default parameters"
            tradeoffs = {}
            confidence = 0.0

        return ParameterRecommendation(
            tool=tool,
            parameters=params,
            reasoning=reasoning,
            expected_tradeoffs=tradeoffs,
            confidence=confidence
        )

    def recommend_batch(
        self,
        tool: str,
        sample_dirs: List[str],
        database_info: Optional[Dict[str, Any]] = None
    ) -> List[ParameterRecommendation]:
        """
        Get parameter recommendations for multiple samples.

        Args:
            tool: Tool name
            sample_dirs: List of sample data directories
            database_info: Database information

        Returns:
            List of ParameterRecommendation objects
        """
        recommendations = []

        for sample_dir in sample_dirs:
            logger.info(f"Processing sample: {sample_dir}")

            # Extract sample features
            sample_features = get_sample_features(sample_dir)

            # Get recommendation
            rec = self.recommend(tool, sample_features, database_info)
            recommendations.append(rec)

        return recommendations

    def get_default_parameters(self, tool: str) -> Dict[str, float]:
        """Get default parameters for a tool."""
        defaults = {}
        for param, info in self.parameter_ranges.get(tool, {}).items():
            defaults[param] = info.get('default', 0)
        return defaults

    def save_recommendations(
        self,
        recommendations: List[ParameterRecommendation],
        output_path: str
    ) -> None:
        """Save recommendations to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for rec in recommendations:
            data.append({
                'tool': rec.tool,
                'parameters': rec.parameters,
                'reasoning': rec.reasoning,
                'expected_tradeoffs': rec.expected_tradeoffs,
                'confidence': rec.confidence
            })

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(recommendations)} recommendations to {output_path}")


def main():
    """CLI entry point."""
    import click

    @click.command()
    @click.option('--config', '-c', default='config/llm_config.yaml', help='Path to config file')
    @click.option('--tool', '-t', type=click.Choice(['kraken2', 'centrifuge', 'pathseq']), required=True)
    @click.option('--input', '-i', 'input_dir', required=True, help='Input sample directory')
    @click.option('--output', '-o', required=True, help='Output recommendations file')
    @click.option('--provider', '-p', default='anthropic', help='LLM provider')
    @click.option('--model', '-m', default='claude-sonnet-4-5-20250929', help='Model name')
    def recommend_params(config, tool, input_dir, output, provider, model):
        """Generate parameter recommendations for a sample."""
        recommender = LLMRecommender(
            config_path=config,
            provider=provider,
            model=model
        )

        sample_features = get_sample_features(input_dir)

        rec = recommender.recommend(tool, sample_features)

        recommender.save_recommendations([rec], output)

        click.echo(f"Recommendations saved to {output}")
        click.echo(f"\nTool: {rec.tool}")
        click.echo(f"Parameters: {rec.parameters}")
        click.echo(f"\nReasoning: {rec.reasoning}")

    recommend_params()


if __name__ == '__main__':
    main()
