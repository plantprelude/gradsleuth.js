"""
Abstract base class for vector stores
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


class BaseVectorStore(ABC):
    """
    Abstract base class for all vector store implementations
    """

    @abstractmethod
    def create_index(
        self,
        index_name: str,
        dimension: int = 768,
        metric: str = 'cosine'
    ) -> None:
        """
        Create a new index

        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dot_product)
        """
        pass

    @abstractmethod
    def insert_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        index_name: str,
        batch_size: int = 1000
    ) -> None:
        """
        Insert vectors with metadata

        Args:
            vectors: Array of shape (n, dimension)
            metadata: List of metadata dicts
            index_name: Index to insert into
            batch_size: Batch size for insertion
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        index_name: str,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            index_name: Index to search
            k: Number of results
            filters: Metadata filters

        Returns:
            List of results with scores and metadata
        """
        pass

    @abstractmethod
    def batch_search(
        self,
        query_vectors: np.ndarray,
        index_name: str,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch search for multiple queries

        Args:
            query_vectors: Array of query embeddings (n_queries, dimension)
            index_name: Index to search
            k: Number of results per query
            filters: Metadata filters

        Returns:
            List of result lists
        """
        pass

    @abstractmethod
    def delete_vectors(
        self,
        ids: List[str],
        index_name: str
    ) -> int:
        """
        Delete vectors by IDs

        Args:
            ids: List of vector IDs
            index_name: Index name

        Returns:
            Number of vectors deleted
        """
        pass

    @abstractmethod
    def update_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        index_name: str
    ) -> bool:
        """
        Update a vector and its metadata

        Args:
            vector_id: ID of vector to update
            vector: New vector
            metadata: New metadata
            index_name: Index name

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def get_vector(
        self,
        vector_id: str,
        index_name: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a vector by ID

        Args:
            vector_id: Vector ID
            index_name: Index name

        Returns:
            Tuple of (vector, metadata) or None
        """
        pass

    @abstractmethod
    def count(self, index_name: str) -> int:
        """
        Get number of vectors in index

        Args:
            index_name: Index name

        Returns:
            Vector count
        """
        pass

    @abstractmethod
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an entire index

        Args:
            index_name: Index to delete

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def list_indices(self) -> List[str]:
        """
        List all available indices

        Returns:
            List of index names
        """
        pass

    @abstractmethod
    def save_index(self, index_name: str, path: str) -> bool:
        """
        Persist index to disk

        Args:
            index_name: Index to save
            path: File path

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def load_index(self, index_name: str, path: str) -> bool:
        """
        Load index from disk

        Args:
            index_name: Name for loaded index
            path: File path

        Returns:
            Success status
        """
        pass
