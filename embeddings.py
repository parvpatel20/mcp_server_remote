"""
Embedding model for converting text to vectors.
Uses LangChain's HuggingFaceEndpointEmbeddings for serverless embeddings.

This replaces the custom wrapper with LangChain's built-in integration.
"""
import asyncio
import logging
import os
from typing import List, Union

from langchain_huggingface import HuggingFaceEndpointEmbeddings

logger = logging.getLogger(__name__)

# Default model — BAAI/bge-large-en-v1.5 produces 1024-dimensional embeddings
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_DIMENSION = 1024


class EmbeddingModel:
    """
    Wrapper around LangChain's HuggingFaceEndpointEmbeddings.
    Maintains backward compatibility with existing interface.
    """

    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: HuggingFace model ID (e.g. "BAAI/bge-large-en-v1.5").
                        Falls back to env var HF_EMBEDDING_MODEL, then DEFAULT_MODEL.
        """
        self.model_name = (
            model_name
            or os.getenv("HF_EMBEDDING_MODEL", DEFAULT_MODEL)
        )
        self._embedding_model = None
        self._ready = False

    def load_model(self) -> None:
        """Initialize LangChain's HuggingFaceEndpointEmbeddings."""
        if self._ready:
            return

        try:
            logger.info(
                f"Initializing LangChain HuggingFaceEndpointEmbeddings: {self.model_name}"
            )
            
            hf_token = os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
            
            self._embedding_model = HuggingFaceEndpointEmbeddings(
                model=self.model_name,
                huggingfacehub_api_token=hf_token,
            )
            
            self._ready = True
            logger.info(f"LangChain embeddings ready (model={self.model_name})")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain embeddings: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """
        Encode text(s) into embedding vectors via LangChain.

        Args:
            texts: Single string or list of strings to embed.
            batch_size: Ignored (kept for backward compat).
            show_progress_bar: Ignored (kept for backward compat).

        Returns:
            List of embedding vectors (list[list[float]]).
        """
        if not self._ready or self._embedding_model is None:
            raise RuntimeError(
                "Embedding model not initialized. Call load_model() first."
            )

        try:
            single_input = isinstance(texts, str)
            if single_input:
                texts = [texts]

            # Use LangChain's embed_documents
            embeddings = self._embedding_model.embed_documents(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode texts via LangChain: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents (sync LangChain interface)."""
        return self.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text (LangChain standard interface).
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector as list[float].
        """
        if not self._ready or self._embedding_model is None:
            raise RuntimeError(
                "Embedding model not initialized. Call load_model() first."
            )
        
        return self._embedding_model.embed_query(text)

    async def aencode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> List[List[float]]:
        """Async encode text(s), using native async method when available."""
        del batch_size, show_progress_bar  # kept for backward compatibility

        if not self._ready or self._embedding_model is None:
            raise RuntimeError(
                "Embedding model not initialized. Call load_model() first."
            )

        if isinstance(texts, str):
            texts = [texts]

        if hasattr(self._embedding_model, "aembed_documents"):
            return await self._embedding_model.aembed_documents(texts)

        return await asyncio.to_thread(self._embedding_model.embed_documents, texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents asynchronously."""
        return await self.aencode(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query asynchronously."""
        if not self._ready or self._embedding_model is None:
            raise RuntimeError(
                "Embedding model not initialized. Call load_model() first."
            )

        if hasattr(self._embedding_model, "aembed_query"):
            return await self._embedding_model.aembed_query(text)

        return await asyncio.to_thread(self._embedding_model.embed_query, text)

    def get_dimension(self) -> int:
        """Return the embedding dimension for the configured model."""
        return DEFAULT_DIMENSION


# Global singleton
embedding_model = EmbeddingModel()
