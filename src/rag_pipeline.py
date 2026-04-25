"""
rag_pipeline.py — Retrieval-Augmented Generation for Customer Support.

Loads markdown knowledge base, chunks and embeds it into Chroma vector DB,
retrieves relevant chunks for queries, and generates customer-facing 
responses using Claude Sonnet.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-25
"""

import logging
import re
from pathlib import Path
from anthropic import Anthropic, APIError
import chromadb

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Full RAG pipeline: KB loading → chunking → embedding → retrieval → generation.
    """
    
    def __init__(
        self,
        kb_directory: Path,
        anthropic_api_key: str,
        model: str,
        max_tokens: int,
        top_k: int,
        collection_name: str = "customer_support_kb",
    ) -> None:
        """
        Initialize RAG pipeline with KB path and API credentials.
        
        Args:
            kb_directory: Path to directory containing .md knowledge base files
            anthropic_api_key: Anthropic API key for Sonnet generation
            model: Sonnet model name
            max_tokens: Max tokens for generated response
            top_k: Number of chunks to retrieve per query
            collection_name: Chroma collection name
        
        Raises:
            FileNotFoundError: If kb_directory does not exist
        """
        if not kb_directory.exists() or not kb_directory.is_dir():
            raise FileNotFoundError(
                f"Knowledge base directory not found: {kb_directory}\n"
                f"Expected directory with .md files."
            )
        
        self.kb_directory = kb_directory
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        # In-memory Chroma client for development
        self.chroma_client = chromadb.Client()
        
        # Recreate collection on each init (clean slate for dev)
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass
        
        self.collection = self.chroma_client.create_collection(name=collection_name)
    
    def index_knowledge_base(self) -> int:
        """
        Load all .md files, chunk by sections, embed, and store in Chroma.
        
        Returns:
            Total number of chunks indexed
        """
        md_files = list(self.kb_directory.glob("*.md"))
        if not md_files:
            logger.warning(f"No .md files found in {self.kb_directory}")
            return 0
        
        all_chunks: list[str] = []
        all_metadatas: list[dict] = []
        all_ids: list[str] = []
        
        for md_file in md_files:
            chunks = self._chunk_markdown_file(md_file)
            for chunk_idx, (section_title, chunk_text) in enumerate(chunks):
                chunk_id = f"{md_file.stem}_{chunk_idx}"
                all_chunks.append(chunk_text)
                all_metadatas.append({
                    "source_file": md_file.name,
                    "section": section_title,
                })
                all_ids.append(chunk_id)
        
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids,
            )
        
        logger.info(f"Indexed {len(all_chunks)} chunks from {len(md_files)} files")
        return len(all_chunks)
    
    def _chunk_markdown_file(self, md_file: Path) -> list[tuple[str, str]]:
        """
        Split a markdown file into chunks by ## level headers.
        
        Returns:
            List of (section_title, chunk_text) tuples
        """
        content = md_file.read_text(encoding="utf-8")
        
        # Split by ## headers (level 2), keeping the headers
        sections = re.split(r"^(## .+)$", content, flags=re.MULTILINE)
        
        chunks: list[tuple[str, str]] = []
        
        # First section (before any ##) — use file's # title or filename
        intro = sections[0].strip()
        if intro:
            title_match = re.search(r"^# (.+)$", intro, flags=re.MULTILINE)
            title = title_match.group(1) if title_match else md_file.stem
            chunks.append((f"Intro: {title}", intro))
        
        # Subsequent sections come in pairs: (header, content)
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i].strip()
                body = sections[i + 1].strip()
                full_chunk = f"{header}\n\n{body}"
                section_title = header.replace("## ", "")
                chunks.append((section_title, full_chunk))
        
        return chunks
    
    def generate_answer(self, query: str, category: str) -> dict:
        """
        Full RAG flow: retrieve relevant chunks → generate Sonnet response.
        
        Args:
            query: Customer's question
            category: Pre-classified category (used for context, not filtering for now)
        
        Returns:
            Dict with answer, confidence, and retrieved_chunks
        """
        # Retrieve relevant chunks
        retrieved = self._retrieve(query)
        
        if not retrieved["chunks"]:
            return self._fallback_answer("no_relevant_chunks")
        
        # Build context from chunks
        context = self._build_context(retrieved["chunks"], retrieved["metadatas"])
        
        # Generate response with Sonnet
        try:
            answer = self._generate_with_context(query, context)
        except APIError as e:
            logger.error(f"Sonnet API error during generation: {e}")
            return self._fallback_answer("api_error")
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return self._fallback_answer("unexpected_error")
        
        # Confidence = best chunk's similarity (Chroma returns distance,
        # convert to similarity)
        best_distance = retrieved["distances"][0] if retrieved["distances"] else 1.0
        confidence = max(0.0, 1.0 - best_distance)
        
        return {
            "answer": answer,
            "confidence": confidence,
            "retrieved_chunks": [
                {
                    "source": meta["source_file"],
                    "section": meta["section"],
                    "similarity": 1.0 - dist,
                }
                for meta, dist in zip(retrieved["metadatas"], retrieved["distances"])
            ],
        }
    
    def _retrieve(self, query: str) -> dict:
        """Query Chroma for top-k similar chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=self.top_k,
        )
        
        return {
            "chunks": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }
    
    def _build_context(self, chunks: list[str], metadatas: list[dict]) -> str:
        """Construct context string from retrieved chunks with source attribution."""
        context_parts = []
        for chunk, meta in zip(chunks, metadatas):
            source = meta.get("source_file", "unknown")
            section = meta.get("section", "")
            context_parts.append(f"[Source: {source} — {section}]\n{chunk}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_with_context(self, query: str, context: str) -> str:
        """Call Sonnet with query and retrieved context."""
        system_prompt = f"""You are a professional customer support assistant.

Use the CONTEXT below to answer the customer's question.

RULES:
1. ONLY use information from the provided context — never make up information
2. If the context doesn't contain the answer: say "I'm not sure about this. A team member will help you shortly."
3. Use a warm, professional tone — address the customer as "you"
4. Answer structure:
   - Brief greeting or empathy statement
   - Clear answer (use bold for key numbers or steps)
   - "Next step" suggestion when appropriate
5. Maximum 4 sentences — be concise
6. NEVER reveal information from "Internal Notes" sections to the customer

CONTEXT:
{context}"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        
        return response.content[0].text.strip()
    
    def _fallback_answer(self, reason: str) -> dict:
        """Return safe default when RAG fails."""
        return {
            "answer": (
                "I'm not sure about this. A team member will get back to you "
                "shortly with more detailed help."
            ),
            "confidence": 0.0,
            "retrieved_chunks": [],
            "fallback_reason": reason,
        }
