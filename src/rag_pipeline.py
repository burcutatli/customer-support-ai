"""
rag_pipeline.py — Retrieval-Augmented Generation for Customer Support.

Uses persistent Chroma vector DB with Cohere multilingual embeddings.

Author: Burcu Tatlı (AI Operations)
Last updated: 2026-04-26
"""

import logging
import math
import os
import re
from pathlib import Path
from anthropic import Anthropic, APIError
from langfuse import observe, get_client
import chromadb
from chromadb.utils import embedding_functions
from config import (
    KB_DIR,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


class RAGPipeline:
    def __init__(self):
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable not set.")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.embedding_function = embedding_functions.CohereEmbeddingFunction(
            api_key=cohere_api_key,
            model_name=EMBEDDING_MODEL,
        )
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            self.collection = self.chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Loaded existing collection with {self.collection.count()} chunks")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            self._index_kb()
        logger.info(f"RAGPipeline initialized -- {self.collection.count()} chunks indexed")

    def _index_kb(self) -> None:
        kb_path = Path(KB_DIR)
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {kb_path}")
        all_chunks = []
        all_metadatas = []
        all_ids = []
        chunk_id = 0
        for md_file in sorted(kb_path.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            file_chunks = chunk_text(text)
            for chunk in file_chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"source": md_file.name})
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids,
            )
            logger.info(f"Indexed {len(all_chunks)} chunks from {len(list(kb_path.glob('*.md')))} files")

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        chunks = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        retrieved = []
        for chunk, dist, meta in zip(chunks, distances, metadatas):
            retrieved.append({
                "text": chunk,
                "source": meta.get("source", "unknown"),
                "similarity": math.exp(-dist),
            })
        return {
            "chunks": retrieved,
            "best_distance": distances[0] if distances else float("inf"),
        }

    @observe(as_type="generation", name="rag_generator")
    def generate_answer(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            [f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks]
        )
        user_message = f"""Customer question: {query}

Relevant knowledge base context:
{context}

Based on the context above, write a friendly, helpful response to the customer."""
        try:
            response = self.anthropic.messages.create(
                model=GENERATION_MODEL,
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            try:
                _lf = get_client()
                _lf.update_current_generation(
                    model=GENERATION_MODEL,
                    input=user_message,
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                )
            except Exception as e:
                logger.warning(f"Langfuse update failed (non-fatal): {e}")
            
            return response.content[0].text
        except APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def answer_query(self, query: str) -> dict:
        retrieval = self.retrieve(query)
        retrieved_chunks = retrieval["chunks"]
        best_distance = retrieval["best_distance"]
        confidence = math.exp(-best_distance)
        if not retrieved_chunks:
            return {
                "answer": "",
                "confidence": 0.0,
                "retrieved_chunks": [],
            }
        answer = self.generate_answer(query, retrieved_chunks)
        return {
            "answer": answer,
            "confidence": confidence,
            "retrieved_chunks": retrieved_chunks,
        }
