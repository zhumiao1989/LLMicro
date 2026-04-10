"""
RAG-based Knowledge Base for LLMicro

This module provides retrieval-augmented generation (RAG) capabilities
for parameter recommendation by organizing and retrieving evidence from
tool documentation and methodological literature.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG dependencies not installed. Install with: pip install sentence-transformers faiss-cpu")

from .utils.io import load_config


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata."""
    text: str
    source_document: str
    section_path: str
    tool_name: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedEvidence:
    """Retrieved evidence chunk with similarity score."""
    chunk: DocumentChunk
    similarity: float
    rank: int


class RAGKnowledgeBase:
    """
    Retrieval-augmented generation knowledge base for parameter evidence.

    This class organizes external evidence from:
    - Tool documentation (Kraken2, Centrifuge, PathSeq)
    - Methodological literature
    - Parameter descriptions and benchmarks

    Features:
    - Hierarchical chunking with metadata preservation
    - Vector indexing for semantic retrieval
    - Multi-round retrieval for implicit parameter evidence
    """

    def __init__(
        self,
        documents_dir: str = 'data/knowledge',
        index_dir: str = 'data/index',
        embedding_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize the RAG knowledge base.

        Args:
            documents_dir: Directory containing source documents
            index_dir: Directory for storing vector index
            embedding_model: Sentence transformer model for embeddings
        """
        self.documents_dir = Path(documents_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model

        # Storage
        self.chunks: List[DocumentChunk] = []
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []

        # Index
        self.index = None
        self.embedding_model = None

        # Initialize if RAG is available
        if RAG_AVAILABLE:
            self._init_embedding_model()
            self._load_or_build_index()
        else:
            logger.warning("RAG not available. Using LLM internal knowledge only.")

    def _init_embedding_model(self):
        """Initialize the embedding model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _load_or_build_index(self):
        """Load existing index or build new one from documents."""
        index_file = self.index_dir / 'faiss_index.bin'
        metadata_file = self.index_dir / 'chunk_metadata.json'

        if index_file.exists() and metadata_file.exists():
            logger.info(f"Loading existing index from {self.index_dir}")
            self.index = faiss.read_index(str(index_file))
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
            self.chunk_metadata = saved_metadata.get('metadata', [])
            self.chunk_texts = saved_metadata.get('texts', [])
            logger.info(f"Loaded {len(self.chunk_texts)} chunks")
        else:
            logger.info("Building new index from documents...")
            self._build_index()

    def _build_index(self):
        """Build vector index from documents in the knowledge directory."""
        # Load and chunk documents
        self._load_documents()

        if len(self.chunk_texts) == 0:
            logger.warning("No documents found. Index will be empty.")
            return

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(self.chunk_texts)} chunks...")
        embeddings = self.embedding_model.encode(
            self.chunk_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Save index
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / 'faiss_index.bin'))

        with open(self.index_dir / 'chunk_metadata.json', 'w') as f:
            json.dump({
                'metadata': self.chunk_metadata,
                'texts': self.chunk_texts
            }, f, indent=2)

        logger.info(f"Index built and saved to {self.index_dir}")

    def _load_documents(self):
        """Load and chunk documents from the knowledge directory."""
        if not self.documents_dir.exists():
            logger.warning(f"Documents directory not found: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return

        # Find all markdown and text files
        doc_files = list(self.documents_dir.glob('**/*.md')) + \
                    list(self.documents_dir.glob('**/*.txt'))

        for doc_file in doc_files:
            logger.debug(f"Processing document: {doc_file}")
            self._chunk_document(doc_file)

    def _chunk_document(self, doc_path: Path):
        """
        Chunk a document into hierarchical sections.

        Uses a two-stage strategy:
        1. Split at section headers (##, ###, etc.)
        2. Further divide long sections into coherent sub-chunks
        """
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract document name
        doc_name = doc_path.stem

        # Split by section headers
        sections = self._split_by_headers(content)

        for section_title, section_text in sections:
            # Further chunk if section is too long
            sub_chunks = self._chunk_text(section_text, max_length=500)

            for i, sub_chunk in enumerate(sub_chunks):
                chunk_id = f"{doc_name}_{section_title[:20]}_{i}"

                # Clean and normalize text
                cleaned_text = self._clean_chunk_text(sub_chunk)

                if len(cleaned_text.strip()) < 50:
                    continue  # Skip very short chunks

                # Determine tool name from content
                tool_name = self._detect_tool(cleaned_text)

                chunk = DocumentChunk(
                    text=cleaned_text,
                    source_document=doc_name,
                    section_path=section_title,
                    tool_name=tool_name,
                    chunk_id=chunk_id,
                    metadata={
                        'length': len(cleaned_text),
                        'file_path': str(doc_path)
                    }
                )

                self.chunks.append(chunk)
                self.chunk_texts.append(cleaned_text)
                self.chunk_metadata.append(chunk.metadata)

    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split text by markdown-style headers."""
        import re

        # Pattern for headers (## Header)
        header_pattern = r'^(#{1,6})\s+(.+)$'

        sections = []
        current_title = "Introduction"
        current_text = []

        for line in text.split('\n'):
            header_match = re.match(header_pattern, line.strip())
            if header_match:
                # Save previous section
                if current_text:
                    sections.append((current_title, '\n'.join(current_text)))

                # Start new section
                current_title = header_match.group(2)
                current_text = []
            else:
                current_text.append(line)

        # Save last section
        if current_text:
            sections.append((current_title, '\n'.join(current_text)))

        return sections

    def _chunk_text(self, text: str, max_length: int = 500) -> List[str]:
        """Split text into chunks of max_length characters."""
        if len(text) <= max_length:
            return [text]

        # Split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            if current_length + para_length > max_length:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length

        # Save last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _clean_chunk_text(self, text: str) -> str:
        """Clean and normalize chunk text."""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix broken words (hyphenation at line breaks)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

        # Remove citation markers like [1], [2, 3]
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)

        # Compress multiple blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _detect_tool(self, text: str) -> str:
        """Detect which tool a chunk refers to."""
        text_lower = text.lower()

        tools = []
        if 'kraken2' in text_lower or 'kraken' in text_lower:
            tools.append('kraken2')
        if 'centrifuge' in text_lower:
            tools.append('centrifuge')
        if 'pathseq' in text_lower or 'path-seq' in text_lower:
            tools.append('pathseq')

        if len(tools) == 1:
            return tools[0]
        elif len(tools) > 1:
            return 'multiple'
        else:
            return 'general'

    def retrieve(
        self,
        query: str,
        tool_name: str,
        top_k: int = 5
    ) -> List[RetrievedEvidence]:
        """
        Retrieve relevant evidence chunks.

        Args:
            query: Query string
            tool_name: Target tool name
            top_k: Number of top results to return

        Returns:
            List of RetrievedEvidence objects
        """
        if not RAG_AVAILABLE or self.index is None:
            logger.warning("RAG not available. Returning empty evidence.")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        # Search index
        distances, indices = self.index.search(query_embedding, k=top_k * 2)

        # Collect results
        results = []
        rank = 0

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity

            # Filter by tool if specified
            if tool_name != 'general' and chunk.tool_name not in [tool_name, 'multiple', 'general']:
                continue

            evidence = RetrievedEvidence(
                chunk=chunk,
                similarity=similarity,
                rank=rank
            )
            results.append(evidence)
            rank += 1

            if rank >= top_k:
                break

        return results

    def retrieve_multi_round(
        self,
        tool_name: str,
        parameter_names: List[str],
        top_k: int = 3
    ) -> Dict[str, List[RetrievedEvidence]]:
        """
        Multi-round retrieval for implicit parameter evidence.

        Args:
            tool_name: Target tool name
            parameter_names: List of parameter names to search for
            top_k: Number of results per round

        Returns:
            Dictionary mapping parameter names to retrieved evidence
        """
        evidence_dict = {}

        for param_name in parameter_names:
            # Formulate query with parameter synonyms and context
            queries = self._expand_parameter_query(param_name, tool_name)

            all_evidence = []
            for query in queries:
                evidence = self.retrieve(query, tool_name, top_k // len(queries))
                all_evidence.extend(evidence)

            # Deduplicate and rank
            seen_ids = set()
            unique_evidence = []
            for ev in all_evidence:
                if ev.chunk.chunk_id not in seen_ids:
                    seen_ids.add(ev.chunk.chunk_id)
                    unique_evidence.append(ev)

            evidence_dict[param_name] = unique_evidence[:top_k]

        return evidence_dict

    def _expand_parameter_query(
        self,
        parameter_name: str,
        tool_name: str
    ) -> List[str]:
        """Expand parameter name with synonyms and context."""
        # Parameter synonyms
        synonyms = {
            'confidence': ['confidence threshold', 'confidence score', 'classification threshold'],
            'minimum-hit-groups': ['hit groups', 'minimum hits', 'k-mer hits'],
            'k': ['report alignments', 'max alignments', 'distinct alignments'],
            'min-hitlen': ['hit length', 'seed length', 'minimum match'],
            'min-clipped-read-length': ['clipped read', 'read length', 'adapter clipping'],
            'host-min-identity': ['host identity', 'host alignment', 'host removal'],
            'min-score-identity': ['score identity', 'alignment identity', 'microbial identity'],
            'identity-margin': ['identity margin', 'margin', 'multiple matching']
        }

        base_queries = [
            f"{tool_name} {parameter_name}",
            f"{tool_name} parameter {parameter_name}"
        ]

        if parameter_name in synonyms:
            for synonym in synonyms[parameter_name]:
                base_queries.append(f"{tool_name} {synonym}")

        return base_queries

    def format_evidence_for_prompt(
        self,
        evidence: List[RetrievedEvidence]
    ) -> str:
        """Format retrieved evidence for inclusion in LLM prompt."""
        if not evidence:
            return "No specific parameter evidence retrieved."

        formatted = []
        for ev in evidence:
            formatted.append(
                f"[Source: {ev.chunk.source_document} | Section: {ev.chunk.section_path}]\n"
                f"Relevance: {ev.similarity:.2f}\n"
                f"Content: {ev.chunk.text[:300]}..."
            )

        return "\n\n".join(formatted)


def main():
    """CLI entry point for building index."""
    import click

    @click.command()
    @click.option('--documents', '-d', default='data/knowledge', help='Documents directory')
    @click.option('--index', '-i', default='data/index', help='Index directory')
    @click.option('--rebuild', is_flag=True, help='Force rebuild index')
    def build_index(documents, index, rebuild):
        """Build or rebuild the RAG knowledge index."""
        if rebuild and Path(index).exists():
            import shutil
            shutil.rmtree(index)
            logger.info(f"Removed existing index: {index}")

        kb = RAGKnowledgeBase(documents_dir=documents, index_dir=index)
        kb._build_index()

        click.echo(f"Index built with {len(kb.chunks)} chunks")

    build_index()


if __name__ == '__main__':
    main()
