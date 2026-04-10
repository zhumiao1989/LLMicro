"""
Tests for RAG Knowledge Base Module
"""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock RAG dependencies if not available
import sys
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from src.rag_knowledge_base import RAGKnowledgeBase, DocumentChunk, RetrievedEvidence


class TestDocumentChunk:
    """Test cases for DocumentChunk data class."""

    def test_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            text="This is a test chunk about Kraken2 confidence parameter.",
            source_document="kraken2_doc.md",
            section_path="Methods/Parameters",
            tool_name="kraken2",
            chunk_id="test_001"
        )

        assert chunk.text == "This is a test chunk about Kraken2 confidence parameter."
        assert chunk.source_document == "kraken2_doc.md"
        assert chunk.tool_name == "kraken2"

    def test_chunk_with_metadata(self):
        """Test creating a chunk with metadata."""
        chunk = DocumentChunk(
            text="Test content",
            source_document="test.md",
            section_path="Introduction",
            tool_name="general",
            chunk_id="test_002",
            metadata={"length": 12, "page": 1}
        )

        assert chunk.metadata["length"] == 12
        assert chunk.metadata["page"] == 1


class TestRAGKnowledgeBase:
    """Test cases for RAGKnowledgeBase class."""

    @pytest.fixture
    def mock_kb(self, tmp_path):
        """Create a mock knowledge base."""
        documents_dir = tmp_path / "documents"
        index_dir = tmp_path / "index"
        documents_dir.mkdir()
        index_dir.mkdir()

        with patch('src.rag_knowledge_base.RAGKnowledgeBase._init_embedding_model'):
            with patch('src.rag_knowledge_base.RAGKnowledgeBase._load_or_build_index'):
                kb = RAGKnowledgeBase(
                    documents_dir=str(documents_dir),
                    index_dir=str(index_dir)
                )
                kb.chunks = [
                    DocumentChunk(
                        text="Kraken2 confidence parameter controls classification precision.",
                        source_document="kraken2_doc.md",
                        section_path="Parameters",
                        tool_name="kraken2",
                        chunk_id="chunk_001"
                    ),
                    DocumentChunk(
                        text="Centrifuge k parameter specifies max distinct alignments.",
                        source_document="centrifuge_doc.md",
                        section_path="Options",
                        tool_name="centrifuge",
                        chunk_id="chunk_002"
                    ),
                    DocumentChunk(
                        text="PathSeq host-min-identity controls host read removal stringency.",
                        source_document="pathseq_doc.md",
                        section_path="Parameters/Host",
                        tool_name="pathseq",
                        chunk_id="chunk_003"
                    )
                ]
                kb.chunk_texts = [c.text for c in kb.chunks]
                return kb

    def test_initialization(self, mock_kb):
        """Test knowledge base initialization."""
        assert mock_kb is not None
        assert len(mock_kb.chunks) == 3

    def test_detect_tool_kraken2(self, mock_kb):
        """Test tool detection for Kraken2."""
        text = "Kraken2 uses k-mer exact matching for classification."
        tool = mock_kb._detect_tool(text)
        assert tool == "kraken2"

    def test_detect_tool_centrifuge(self, mock_kb):
        """Test tool detection for Centrifuge."""
        text = "Centrifuge provides compressed indexing for metagenomic sequences."
        tool = mock_kb._detect_tool(text)
        assert tool == "centrifuge"

    def test_detect_tool_pathseq(self, mock_kb):
        """Test tool detection for PathSeq."""
        text = "PathSeq integrates host-sequence filtering with microbial alignment."
        tool = mock_kb._detect_tool(text)
        assert tool == "pathseq"

    def test_detect_tool_multiple(self, mock_kb):
        """Test tool detection for multiple tools."""
        text = "Kraken2 and Centrifuge are both metagenomic classifiers."
        tool = mock_kb._detect_tool(text)
        assert tool == "multiple"

    def test_detect_tool_general(self, mock_kb):
        """Test tool detection when no specific tool mentioned."""
        text = "Metagenomic classification requires quality control."
        tool = mock_kb._detect_tool(text)
        assert tool == "general"

    def test_clean_chunk_text(self, mock_kb):
        """Test text cleaning."""
        # Test excessive whitespace removal
        text = "This   has   extra   spaces."
        cleaned = mock_kb._clean_chunk_text(text)
        assert "  " not in cleaned

        # Test citation marker removal
        text = "Kraken2 is fast [1, 2, 3]."
        cleaned = mock_kb._clean_chunk_text(text)
        assert "[" not in cleaned

    def test_split_by_headers(self, mock_kb):
        """Test splitting text by headers."""
        text = """# Introduction
This is the introduction.

## Methods
This is the methods section.

### Parameters
This describes parameters.

## Results
This is results.
"""
        sections = mock_kb._split_by_headers(text)

        assert len(sections) >= 3
        assert any("Introduction" in s[0] for s in sections)
        assert any("Methods" in s[0] for s in sections)

    def test_chunk_text(self, mock_kb):
        """Test text chunking."""
        # Short text (should not be split)
        text = "Short text."
        chunks = mock_kb._chunk_text(text, max_length=100)
        assert len(chunks) == 1

        # Long text (should be split)
        long_text = "A. " * 100  # 200 chars
        chunks = mock_kb._chunk_text(long_text, max_length=100)
        assert len(chunks) > 1

    def test_expand_parameter_query(self, mock_kb):
        """Test query expansion for parameter retrieval."""
        queries = mock_kb._expand_parameter_query("confidence", "kraken2")

        assert len(queries) > 1
        assert any("kraken2" in q.lower() for q in queries)
        assert any("confidence" in q.lower() for q in queries)

    def test_format_evidence_for_prompt(self, mock_kb):
        """Test formatting evidence for LLM prompt."""
        evidence = [
            RetrievedEvidence(
                chunk=mock_kb.chunks[0],
                similarity=0.95,
                rank=0
            )
        ]

        formatted = mock_kb.format_evidence_for_prompt(evidence)

        assert "kraken2_doc.md" in formatted
        assert "0.95" in formatted
        assert "Kraken2" in formatted

    def test_format_evidence_empty(self, mock_kb):
        """Test formatting empty evidence."""
        formatted = mock_kb.format_evidence_for_prompt([])
        assert "No specific parameter evidence" in formatted


class TestRetrievedEvidence:
    """Test cases for RetrievedEvidence data class."""

    def test_evidence_creation(self):
        """Test creating retrieved evidence."""
        chunk = DocumentChunk(
            text="Test content",
            source_document="test.md",
            section_path="Methods",
            tool_name="kraken2",
            chunk_id="test"
        )

        evidence = RetrievedEvidence(
            chunk=chunk,
            similarity=0.85,
            rank=1
        )

        assert evidence.chunk == chunk
        assert evidence.similarity == 0.85
        assert evidence.rank == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
