"""
LLM Prompts for parameter recommendation

LLMicro: Context-aware parameter recommendation with RAG-based evidence retrieval.
"""

from typing import Dict, Any, Optional, List


def get_parameter_recommendation_prompt(
    tool: str,
    sample_features: Dict[str, Any],
    database_info: Dict[str, Any],
    resource_constraints: Dict[str, Any],
    parameter_ranges: Dict[str, Any],
    retrieved_evidence: Optional[Dict[str, List]] = None
) -> str:
    """
    Generate prompt for LLM parameter recommendation.

    Args:
        tool: Classification tool name (kraken2, centrifuge, pathseq)
        sample_features: Dictionary with sample characteristics
        database_info: Database information
        resource_constraints: Resource constraints
        parameter_ranges: Valid parameter ranges
        retrieved_evidence: Optional dictionary of retrieved RAG evidence

    Returns:
        Formatted prompt string
    """

    prompt = f"""You are an expert bioinformatics assistant specializing in metagenomic classification parameter optimization.

## Task
Recommend optimal parameters for **{tool.upper()}** based on the provided sample characteristics and retrieved evidence.

## Sample Features
- **Sequencing depth**: {sample_features.get('sequencing_depth', 'Unknown')} reads
- **Mean read length**: {sample_features.get('mean_read_length', 'Unknown')} bp
- **Mean quality score**: {sample_features.get('mean_quality', 'Unknown')} (Phred+33)
- **Estimated complexity (Shannon diversity)**: {sample_features.get('estimated_complexity', 'Unknown')}

## Database Information
- **Source**: {database_info.get('source', 'Unknown')}
- **Taxa covered**: {', '.join(database_info.get('taxa', []))}

## Resource Constraints
- **Maximum memory**: {resource_constraints.get('max_memory_gb', 64)} GB
- **Maximum threads**: {resource_constraints.get('max_threads', 16)}
- **Target speed**: {resource_constraints.get('target_speed_m_reads_per_min', 1.0)} M reads/min
"""

    # Add retrieved evidence section (RAG)
    if retrieved_evidence:
        prompt += "\n## Retrieved Evidence from Knowledge Base\n\n"
        for param_name, evidence_list in retrieved_evidence.items():
            if evidence_list:
                prompt += f"**For parameter '{param_name}':**\n"
                for ev in evidence_list[:3]:  # Top 3 evidence
                    prompt += f"- [Source: {ev.chunk.source_document}] {ev.chunk.text[:200]}... (relevance: {ev.similarity:.2f})\n"
                prompt += "\n"

    prompt += "\n## Available Parameters and Ranges\n"

    # Add parameter ranges for the specific tool
    tool_ranges = parameter_ranges.get(tool, {})
    for param_name, param_info in tool_ranges.items():
        prompt += f"- **{param_name}**: {param_info.get('min', 'N/A')} to {param_info.get('max', 'N/A')} (default: {param_info.get('default', 'N/A')})\n"
        prompt += f"  - Description: {param_info.get('description', 'No description')}\n"

    prompt += """
## Optimization Goals
Balance the following objectives:
1. **Precision**: Minimize false positives (especially important for low-abundance taxa)
2. **Recall**: Maximize true detections
3. **Abundance estimation accuracy**: Minimize L1/L2 error
4. **Computational efficiency**: Stay within resource constraints

## Structured Output Constraints
- Only output valid parameter names from the Allowed Parameters list
- Do not generate undefined parameters or out-of-range values
- For parameters without sufficient evidence support, retain default values
- All values must be within the specified min-max ranges

## Output Format
Provide your recommendation in the following JSON format:

```json
{
    "recommended_parameters": {
        "param1": value1,
        "param2": value2
    },
    "reasoning": "Brief explanation of your choices, referencing retrieved evidence where applicable",
    "expected_tradeoffs": {
        "precision": "increase/decrease/stable",
        "recall": "increase/decrease/stable",
        "speed": "faster/slower/similar",
        "memory": "higher/lower/similar"
    },
    "confidence": 0.0-1.0,
    "evidence_used": ["list of parameter names that had retrieved evidence support"]
}
```

## Recommendation
"""

    return prompt


def get_comparison_prompt(
    tool: str,
    default_results: Dict[str, Any],
    llm_results: Dict[str, Any]
) -> str:
    """
    Generate prompt for comparing default vs LLM-recommended results.

    Args:
        tool: Classification tool name
        default_results: Results with default parameters
        llm_results: Results with LLM-recommended parameters

    Returns:
        Formatted prompt string
    """

    prompt = f"""You are an expert bioinformatics analyst comparing metagenomic classification results.

## Task
Compare the results of {tool.upper()} with default parameters vs LLM-recommended parameters.

## Default Parameters Results
"""

    for metric, value in default_results.items():
        prompt += f"- **{metric}**: {value}\n"

    prompt += "\n## LLM-Recommended Parameters Results\n"

    for metric, value in llm_results.items():
        prompt += f"- **{metric}**: {value}\n"

    prompt += """
## Analysis
Provide a brief analysis of:
1. Which metrics improved and why
2. Any trade-offs observed
3. Recommendations for similar samples in the future

## Analysis
"""

    return prompt


def get_batch_recommendation_prompt(
    tool: str,
    samples: list,
    parameter_ranges: Dict[str, Any]
) -> str:
    """
    Generate prompt for batch parameter recommendation across multiple samples.

    Args:
        tool: Classification tool name
        samples: List of sample feature dictionaries
        parameter_ranges: Valid parameter ranges

    Returns:
        Formatted prompt string
    """

    prompt = f"""You are an expert bioinformatics assistant. Recommend parameters for **{tool.upper()}** across multiple samples.

## Samples Summary
- **Number of samples**: {len(samples)}
- **Average sequencing depth**: {sum(s.get('sequencing_depth', 0) for s in samples) / len(samples):.0f} reads
- **Average complexity**: {sum(s.get('estimated_complexity', 0) for s in samples) / len(samples):.2f}

## Parameter Ranges
"""

    for param_name, param_info in parameter_ranges.get(tool, {}).items():
        prompt += f"- **{param_name}**: {param_info.get('min', 'N/A')} to {param_info.get('max', 'N/A')}\n"

    prompt += """
## Task
Provide a single recommended parameter configuration that works well across this batch of samples.
Consider the average characteristics while ensuring robustness across sample variability.

## Output Format
```json
{
    "recommended_parameters": { ... },
    "reasoning": "...",
    "robustness_notes": "How this configuration handles sample variability"
}
```

## Recommendation
"""

    return prompt


# Pre-built prompt templates for common scenarios
SCENARIO_PROMPTS = {
    "low_complexity": """
## Scenario: Low Complexity Sample
This sample has low microbial diversity (10-30 species) with relatively uniform abundance distribution.

**Key considerations:**
- False positives are easier to identify due to low expected diversity
- Priority: High precision while maintaining good recall
- Can use more stringent parameters without significant sensitivity loss
""",

    "medium_complexity": """
## Scenario: Medium Complexity Sample
This sample has moderate microbial diversity (50-150 species) with lognormal abundance distribution.

**Key considerations:**
- Balance between precision and recall is critical
- Abundance estimation accuracy is important
- Some rare species may be present
""",

    "high_complexity": """
## Scenario: High Complexity Sample
This sample has high microbial diversity (200-400 species) with stepped/long-tail abundance distribution.

**Key considerations:**
- False positive control is paramount due to long-tail noise
- Low-abundance species detection is challenging
- Prioritize specificity while maintaining reasonable sensitivity
""",

    "high_host": """
## Scenario: High Host Contamination
This sample has high host DNA content (>90% human reads).

**Key considerations:**
- Host removal stringency is critical
- Microbial signal is compressed, requiring careful parameter tuning
- False positives from host-microbe homology regions are a concern
""",

    "low_depth": """
## Scenario: Low Sequencing Depth
This sample has low sequencing depth (<1M microbial reads).

**Key considerations:**
- Statistical power is limited
- Avoid overly stringent parameters that may miss true signals
- Accept higher uncertainty in abundance estimates
""",

    "high_depth": """
## Scenario: High Sequencing Depth
This sample has high sequencing depth (>10M microbial reads).

**Key considerations:**
- Sufficient power for stringent filtering
- Can afford higher confidence thresholds
- Focus on precision to control false discovery
"""
}


def get_scenario_enhanced_prompt(
    tool: str,
    sample_features: Dict[str, Any],
    scenario: str,
    parameter_ranges: Dict[str, Any]
) -> str:
    """
    Generate prompt enhanced with scenario-specific guidance.

    Args:
        tool: Classification tool name
        sample_features: Sample characteristics
        scenario: Scenario name (low_complexity, medium_complexity, etc.)
        parameter_ranges: Valid parameter ranges

    Returns:
        Formatted prompt string
    """

    base_prompt = get_parameter_recommendation_prompt(
        tool, sample_features, {}, {}, parameter_ranges
    )

    scenario_hint = SCENARIO_PROMPTS.get(scenario, "")

    return base_prompt + scenario_hint + "\n\n## Recommendation\n"
