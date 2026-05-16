"""
Pinecone vector database client.

Uses native Pinecone SDK directly (langchain-pinecone not available for Python 3.14).
Provides compatibility layer for future LangChain integration.
"""
import logging
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec

from settings import settings

logger = logging.getLogger(__name__)


class PineconeClient:
    """
    Client for interacting with Pinecone vector database.
    Uses native Pinecone SDK directly.
    """

    def __init__(self):
        """Initialize Pinecone client."""
        self.pc = None
        self.index_name = None
        self.dimension = settings.pinecone_dimension  # BAAI/bge-large-en-v1.5 embedding dimension
        self.index = None

    def initialize_index(self) -> None:
        """Create or connect to Pinecone index."""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            self.index_name = settings.pinecone_index_name

            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]

            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Connecting to existing index: {self.index_name}")

            # Get index reference
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            raise

    def upsert_vectors(
        self,
        vectors: List[tuple],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone index.

        Args:
            vectors: List of (id, embedding, metadata) tuples
            namespace: Optional namespace for organizing vectors

        Returns:
            Upsert response from Pinecone
        """
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")

        try:
            response = self.index.upsert(
                vectors=vectors,
                namespace=namespace
            )
            logger.info(f"Upserted {len(vectors)} vectors to namespace '{namespace}'")
            return response
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query Pinecone index for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            namespace: Namespace to query
            filter: Metadata filter
            include_metadata: Whether to include metadata in results

        Returns:
            Query results with matches
        """
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")

        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=include_metadata
            )
            logger.debug(f"Query returned {len(results.matches)} matches")
            return results
        except Exception as e:
            logger.error(f"Failed to query vectors: {e}")
            raise

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = ""
    ) -> None:
        """
        Delete vectors from index.

        Args:
            ids: List of vector IDs to delete
            delete_all: Delete all vectors in namespace
            namespace: Namespace to delete from
        """
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")

        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Deleted all vectors from namespace '{namespace}'")
            elif ids:
                self.index.delete(ids=ids, namespace=namespace)
                logger.info(f"Deleted {len(ids)} vectors from namespace '{namespace}'")
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.index:
            raise RuntimeError("Index not initialized. Call initialize_index() first.")

        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise


# Global instance
pinecone_client = PineconeClient()
