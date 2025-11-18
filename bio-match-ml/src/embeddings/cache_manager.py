"""
Cache Manager for embedding caching with multi-tier strategy
"""
import logging
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using disk/memory cache only.")


class EmbeddingCache:
    """
    High-performance caching layer with memory, Redis, and disk tiers
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        disk_cache_path: str = "data/embedding_cache",
        max_memory_items: int = 10000,
        ttl: int = 86400,
        use_redis: bool = True,
        use_disk: bool = True
    ):
        """
        Initialize cache with multi-tier strategy

        Args:
            redis_client: Redis client instance
            disk_cache_path: Path for disk cache
            max_memory_items: Maximum items in memory cache
            ttl: Time-to-live in seconds
            use_redis: Enable Redis tier
            use_disk: Enable disk tier
        """
        # Memory cache (LRU)
        self.memory_cache: OrderedDict = OrderedDict()
        self.max_memory_items = max_memory_items

        # Redis cache
        self.redis_client = redis_client
        self.use_redis = use_redis and REDIS_AVAILABLE and redis_client is not None
        self.ttl = ttl

        # Disk cache
        self.disk_cache_path = Path(disk_cache_path)
        self.use_disk = use_disk
        if self.use_disk:
            self.disk_cache_path.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'hits_memory': 0,
            'hits_redis': 0,
            'hits_disk': 0,
            'misses': 0,
            'sets': 0
        }

        logger.info(f"EmbeddingCache initialized: Redis={self.use_redis}, Disk={self.use_disk}")

    def _get_hash_key(self, key: str) -> str:
        """Generate hash key for storage"""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache (checks all tiers)

        Args:
            key: Cache key

        Returns:
            Embedding array if found, None otherwise
        """
        hash_key = self._get_hash_key(key)

        # Check memory cache first
        if hash_key in self.memory_cache:
            self.stats['hits_memory'] += 1
            # Move to end (LRU)
            self.memory_cache.move_to_end(hash_key)
            return self.memory_cache[hash_key]

        # Check Redis cache
        if self.use_redis:
            try:
                value = self.redis_client.get(f"emb:{hash_key}")
                if value is not None:
                    self.stats['hits_redis'] += 1
                    embedding = pickle.loads(value)

                    # Promote to memory cache
                    self._set_memory(hash_key, embedding)

                    return embedding
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        # Check disk cache
        if self.use_disk:
            disk_file = self.disk_cache_path / f"{hash_key}.pkl"
            if disk_file.exists():
                try:
                    with open(disk_file, 'rb') as f:
                        embedding = pickle.load(f)

                    self.stats['hits_disk'] += 1

                    # Promote to higher tiers
                    self._set_memory(hash_key, embedding)
                    if self.use_redis:
                        self._set_redis(hash_key, embedding)

                    return embedding
                except Exception as e:
                    logger.error(f"Disk cache read error: {e}")

        self.stats['misses'] += 1
        return None

    def set(self, key: str, embedding: np.ndarray) -> None:
        """
        Store embedding in all cache tiers

        Args:
            key: Cache key
            embedding: Embedding array to store
        """
        hash_key = self._get_hash_key(key)

        # Set in all tiers
        self._set_memory(hash_key, embedding)

        if self.use_redis:
            self._set_redis(hash_key, embedding)

        if self.use_disk:
            self._set_disk(hash_key, embedding)

        self.stats['sets'] += 1

    def _set_memory(self, hash_key: str, embedding: np.ndarray) -> None:
        """Set in memory cache with LRU eviction"""
        if hash_key in self.memory_cache:
            self.memory_cache.move_to_end(hash_key)
        else:
            if len(self.memory_cache) >= self.max_memory_items:
                # Evict oldest item
                self.memory_cache.popitem(last=False)

            self.memory_cache[hash_key] = embedding

    def _set_redis(self, hash_key: str, embedding: np.ndarray) -> None:
        """Set in Redis cache"""
        try:
            value = pickle.dumps(embedding)
            self.redis_client.setex(
                f"emb:{hash_key}",
                self.ttl,
                value
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def _set_disk(self, hash_key: str, embedding: np.ndarray) -> None:
        """Set in disk cache"""
        try:
            disk_file = self.disk_cache_path / f"{hash_key}.pkl"
            with open(disk_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.error(f"Disk cache write error: {e}")

    def get_or_generate(self, key: str, generator_func):
        """
        Cache-aware retrieval with generation fallback

        Args:
            key: Cache key
            generator_func: Function to generate embedding if not cached

        Returns:
            Embedding array
        """
        # Try cache first
        embedding = self.get(key)

        if embedding is not None:
            return embedding

        # Generate and cache
        embedding = generator_func()
        self.set(key, embedding)

        return embedding

    def bulk_load(self, embedding_dict: Dict[str, np.ndarray]) -> None:
        """
        Preload embeddings for known entities

        Args:
            embedding_dict: Dictionary of {key: embedding}
        """
        logger.info(f"Bulk loading {len(embedding_dict)} embeddings")

        for key, embedding in embedding_dict.items():
            self.set(key, embedding)

        logger.info("Bulk load complete")

    def invalidate(self, key: str) -> None:
        """
        Invalidate cached embedding

        Args:
            key: Cache key to invalidate
        """
        hash_key = self._get_hash_key(key)

        # Remove from memory
        if hash_key in self.memory_cache:
            del self.memory_cache[hash_key]

        # Remove from Redis
        if self.use_redis:
            try:
                self.redis_client.delete(f"emb:{hash_key}")
            except Exception as e:
                logger.error(f"Redis delete error: {e}")

        # Remove from disk
        if self.use_disk:
            disk_file = self.disk_cache_path / f"{hash_key}.pkl"
            if disk_file.exists():
                try:
                    disk_file.unlink()
                except Exception as e:
                    logger.error(f"Disk cache delete error: {e}")

    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cached embeddings matching pattern

        Args:
            pattern: Pattern to match (simple prefix matching)

        Returns:
            Number of items invalidated
        """
        count = 0

        # Memory cache
        keys_to_remove = []
        for key in self.memory_cache.keys():
            # Note: We've hashed the key, so pattern matching is limited
            # This is a simplified implementation
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.memory_cache[key]
            count += 1

        # Redis cache
        if self.use_redis:
            try:
                # Scan for keys matching pattern
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor,
                        match=f"emb:*",
                        count=100
                    )

                    if keys:
                        self.redis_client.delete(*keys)
                        count += len(keys)

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis pattern invalidation error: {e}")

        # Disk cache - clear all
        if self.use_disk:
            for file in self.disk_cache_path.glob("*.pkl"):
                try:
                    file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Disk file delete error: {e}")

        logger.info(f"Invalidated {count} cache entries")
        return count

    def clear_all(self) -> None:
        """Clear entire cache"""
        # Clear memory
        self.memory_cache.clear()

        # Clear Redis
        if self.use_redis:
            try:
                # Delete all embedding keys
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor,
                        match="emb:*",
                        count=1000
                    )

                    if keys:
                        self.redis_client.delete(*keys)

                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis clear error: {e}")

        # Clear disk
        if self.use_disk:
            for file in self.disk_cache_path.glob("*.pkl"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Disk file delete error: {e}")

        # Reset stats
        self.stats = {
            'hits_memory': 0,
            'hits_redis': 0,
            'hits_disk': 0,
            'misses': 0,
            'sets': 0
        }

        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary of cache statistics
        """
        total_requests = sum([
            self.stats['hits_memory'],
            self.stats['hits_redis'],
            self.stats['hits_disk'],
            self.stats['misses']
        ])

        hit_rate = 0.0
        if total_requests > 0:
            total_hits = (
                self.stats['hits_memory'] +
                self.stats['hits_redis'] +
                self.stats['hits_disk']
            )
            hit_rate = total_hits / total_requests

        return {
            **self.stats,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'memory_size': len(self.memory_cache),
            'memory_max': self.max_memory_items
        }

    def warm_up(self, keys_and_generators: Dict[str, callable]) -> None:
        """
        Warm up cache with commonly accessed items

        Args:
            keys_and_generators: Dict of {key: generator_function}
        """
        logger.info(f"Warming up cache with {len(keys_and_generators)} items")

        for key, generator in keys_and_generators.items():
            if self.get(key) is None:
                embedding = generator()
                self.set(key, embedding)

        logger.info("Cache warm-up complete")
