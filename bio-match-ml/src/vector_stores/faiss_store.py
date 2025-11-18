"""
FAISS-based vector store implementation
"""
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.error("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

from .base_store import BaseVectorStore


class FaissStore(BaseVectorStore):
    """
    High-performance local vector store using FAISS
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize FAISS store

        Args:
            use_gpu: Use GPU acceleration if available
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required but not installed")

        self.indices: Dict[str, faiss.Index] = {}
        self.id_mappings: Dict[str, Dict] = {}  # Maps internal ID to external ID
        self.metadata_store: Dict[str, Dict] = {}  # Stores metadata per index
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        if self.use_gpu:
            logger.info(f"FAISS GPU support enabled. GPUs available: {faiss.get_num_gpus()}")
        else:
            logger.info("FAISS running on CPU")

    def create_index(
        self,
        index_name: str,
        dimension: int = 768,
        metric: str = 'cosine',
        index_type: str = 'Flat'
    ) -> None:
        """
        Create a new FAISS index

        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric: Distance metric
            index_type: FAISS index type (Flat, IVF, HNSW, etc.)
        """
        # Normalize vectors for cosine similarity
        normalize_vectors = metric == 'cosine'

        # Create index based on type
        if index_type == 'Flat':
            if metric == 'cosine':
                index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
            else:
                index = faiss.IndexFlatL2(dimension)

        elif index_type == 'IVF':
            # Inverted file index for faster search on large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = 100  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        elif index_type == 'IVF_PQ':
            # IVF with product quantization for memory efficiency
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = 100
            m = 8  # Number of sub-quantizers
            nbits = 8  # Bits per sub-quantizer
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

        elif index_type == 'HNSW':
            # Hierarchical Navigable Small World graph
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(dimension, M)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.indices[index_name] = index
        self.id_mappings[index_name] = {'internal_to_external': {}, 'external_to_internal': {}}
        self.metadata_store[index_name] = {}

        logger.info(f"Created FAISS index '{index_name}' (type={index_type}, dim={dimension})")

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
            metadata: List of metadata dicts (must include 'id' field)
            index_name: Index to insert into
            batch_size: Batch size for insertion
        """
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' does not exist")

        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata items")

        index = self.indices[index_name]
        id_mapping = self.id_mappings[index_name]
        metadata_store = self.metadata_store[index_name]

        # Normalize vectors for cosine similarity
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)

        # Get starting internal ID
        start_id = index.ntotal

        # Train index if needed (for IVF indices)
        if hasattr(index, 'is_trained') and not index.is_trained:
            logger.info(f"Training index '{index_name}' on {len(vectors)} vectors")
            index.train(vectors)

        # Add vectors in batches
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            # Add to index
            index.add(batch_vectors)

            # Store metadata and ID mappings
            for j, meta in enumerate(batch_metadata):
                internal_id = start_id + i + j
                external_id = meta.get('id', str(internal_id))

                id_mapping['internal_to_external'][internal_id] = external_id
                id_mapping['external_to_internal'][external_id] = internal_id
                metadata_store[external_id] = meta

        logger.info(f"Inserted {len(vectors)} vectors into '{index_name}'")

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
            filters: Metadata filters (applied post-search)

        Returns:
            List of results with scores and metadata
        """
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' does not exist")

        index = self.indices[index_name]
        id_mapping = self.id_mappings[index_name]
        metadata_store = self.metadata_store[index_name]

        # Normalize query vector
        query_vector = query_vector.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search
        # Fetch more if filtering is needed
        search_k = k * 10 if filters else k
        scores, indices = index.search(query_vector, search_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue

            internal_id = int(idx)
            external_id = id_mapping['internal_to_external'].get(internal_id)

            if external_id is None:
                continue

            metadata = metadata_store.get(external_id, {})

            # Apply filters if provided
            if filters:
                if not self._matches_filters(metadata, filters):
                    continue

            results.append({
                'id': external_id,
                'score': float(score),
                'metadata': metadata
            })

            # Stop if we have enough results
            if len(results) >= k:
                break

        return results

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
        # For simplicity, use individual searches
        # Can be optimized for batch processing
        results = []
        for query_vector in query_vectors:
            query_results = self.search(query_vector, index_name, k, filters)
            results.append(query_results)

        return results

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filters.items():
            if key not in metadata:
                return False

            meta_value = metadata[key]

            # Handle different filter types
            if isinstance(value, dict):
                # Range filters: {'gte': 10, 'lte': 100}
                if 'gte' in value and meta_value < value['gte']:
                    return False
                if 'lte' in value and meta_value > value['lte']:
                    return False
                if 'gt' in value and meta_value <= value['gt']:
                    return False
                if 'lt' in value and meta_value >= value['lt']:
                    return False
            elif isinstance(value, list):
                # Value must be in list
                if meta_value not in value:
                    return False
            else:
                # Exact match
                if meta_value != value:
                    return False

        return True

    def delete_vectors(
        self,
        ids: List[str],
        index_name: str
    ) -> int:
        """
        Delete vectors by IDs

        Note: FAISS doesn't support efficient deletion,
        so we mark as deleted and rebuild on next save
        """
        if index_name not in self.indices:
            raise ValueError(f"Index '{index_name}' does not exist")

        id_mapping = self.id_mappings[index_name]
        metadata_store = self.metadata_store[index_name]

        deleted_count = 0
        for external_id in ids:
            if external_id in metadata_store:
                # Remove from metadata and mappings
                internal_id = id_mapping['external_to_internal'].pop(external_id, None)
                if internal_id is not None:
                    id_mapping['internal_to_external'].pop(internal_id, None)
                metadata_store.pop(external_id, None)
                deleted_count += 1

        logger.info(f"Marked {deleted_count} vectors for deletion in '{index_name}'")
        return deleted_count

    def update_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        index_name: str
    ) -> bool:
        """
        Update a vector and its metadata

        Note: FAISS doesn't support in-place updates efficiently
        """
        # Delete and re-insert
        self.delete_vectors([vector_id], index_name)
        metadata['id'] = vector_id
        self.insert_vectors(
            vector.reshape(1, -1),
            [metadata],
            index_name
        )
        return True

    def get_vector(
        self,
        vector_id: str,
        index_name: str
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a vector by ID

        Note: FAISS doesn't support direct retrieval,
        returns metadata only
        """
        if index_name not in self.indices:
            return None

        metadata_store = self.metadata_store[index_name]
        metadata = metadata_store.get(vector_id)

        if metadata is None:
            return None

        # Can't easily retrieve vector from FAISS
        # Return None for vector, metadata only
        return None, metadata

    def count(self, index_name: str) -> int:
        """Get number of vectors in index"""
        if index_name not in self.indices:
            return 0

        return self.indices[index_name].ntotal

    def delete_index(self, index_name: str) -> bool:
        """Delete an entire index"""
        if index_name in self.indices:
            del self.indices[index_name]
            del self.id_mappings[index_name]
            del self.metadata_store[index_name]
            logger.info(f"Deleted index '{index_name}'")
            return True
        return False

    def list_indices(self) -> List[str]:
        """List all available indices"""
        return list(self.indices.keys())

    def save_index(self, index_name: str, path: str) -> bool:
        """Persist index to disk"""
        if index_name not in self.indices:
            return False

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index = self.indices[index_name]

        # Move to CPU if on GPU
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, str(path_obj))

        # Save metadata and mappings
        metadata_path = path_obj.with_suffix('.metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_mapping': self.id_mappings[index_name],
                'metadata_store': self.metadata_store[index_name]
            }, f)

        logger.info(f"Saved index '{index_name}' to {path}")
        return True

    def load_index(self, index_name: str, path: str) -> bool:
        """Load index from disk"""
        path_obj = Path(path)

        if not path_obj.exists():
            logger.error(f"Index file not found: {path}")
            return False

        # Load FAISS index
        index = faiss.read_index(str(path_obj))

        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self.indices[index_name] = index

        # Load metadata and mappings
        metadata_path = path_obj.with_suffix('.metadata.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.id_mappings[index_name] = data['id_mapping']
                self.metadata_store[index_name] = data['metadata_store']
        else:
            self.id_mappings[index_name] = {'internal_to_external': {}, 'external_to_internal': {}}
            self.metadata_store[index_name] = {}

        logger.info(f"Loaded index '{index_name}' from {path}")
        return True
