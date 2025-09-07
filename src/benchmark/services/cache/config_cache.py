"""
Advanced configuration cache with LRU eviction and performance optimizations.

This module provides memory-efficient caching for configuration objects with:
- LRU (Least Recently Used) eviction policy
- Memory usage monitoring
- Cache statistics and hit rate tracking
- Thread-safe operations
- Configuration diff tracking
"""

import asyncio
import threading
import time
from collections import OrderedDict
from typing import Any

from benchmark.core.config import ExperimentConfig
from benchmark.core.logging import get_logger


class ConfigurationCache:
    """
    LRU cache for configuration objects with advanced features.

    Features:
    - LRU eviction when max_size is reached
    - TTL (Time To Live) for cache entries
    - Memory usage estimation and limits
    - Cache statistics and monitoring
    - Thread-safe operations
    - Weak references for memory efficiency
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        max_memory_mb: int = 256,
        enable_stats: bool = True,
    ):
        """
        Initialize the configuration cache.

        Args:
            max_size: Maximum number of configurations to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            max_memory_mb: Maximum memory usage in MB
            enable_stats: Whether to collect cache statistics
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_stats = enable_stats

        # LRU cache implemented with OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()

        # Statistics
        self._stats = CacheStats() if enable_stats else None

        # Memory tracking
        self._estimated_memory_usage = 0

        # Background cleanup task
        self._cleanup_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

        self.logger = get_logger("config_cache")

    async def initialize(self) -> None:
        """Initialize the cache and start background cleanup."""
        self._shutdown_event.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info(
            f"Configuration cache initialized (max_size={self.max_size}, ttl={self.ttl_seconds}s)"
        )

    async def shutdown(self) -> None:
        """Shutdown the cache and cleanup resources."""
        self._shutdown_event.set()
        if self._cleanup_task:
            await self._cleanup_task

        with self._cache_lock:
            self._cache.clear()
            self._estimated_memory_usage = 0

        self.logger.info("Configuration cache shutdown completed")

    async def get_config(self, config_id: str) -> ExperimentConfig | None:
        """
        Get a configuration from the cache.

        Args:
            config_id: Unique identifier for the configuration

        Returns:
            Cached configuration or None if not found/expired
        """
        with self._cache_lock:
            entry = self._cache.get(config_id)

            if entry is None:
                if self._stats:
                    self._stats.record_miss()
                return None

            # Check if entry is expired
            if self._is_expired(entry):
                self._remove_entry(config_id)
                if self._stats:
                    self._stats.record_miss()
                return None

            # Move to end for LRU ordering
            self._cache.move_to_end(config_id)
            entry.last_accessed = time.time()

            if self._stats:
                self._stats.record_hit()

            self.logger.debug(f"Cache hit for config {config_id}")
            return entry.config

    async def set_config(
        self, config_id: str, config: ExperimentConfig, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Store a configuration in the cache.

        Args:
            config_id: Unique identifier for the configuration
            config: Configuration object to cache
            metadata: Optional metadata for the cache entry
        """
        estimated_size = self._estimate_config_size(config)

        with self._cache_lock:
            # Remove existing entry if present
            if config_id in self._cache:
                self._remove_entry(config_id)

            # Check memory limits
            while (
                self._estimated_memory_usage + estimated_size > self.max_memory_bytes
                and len(self._cache) > 0
            ):
                self._evict_oldest()

            # Check size limits
            while len(self._cache) >= self.max_size:
                self._evict_oldest()

            # Create new entry
            entry = CacheEntry(
                config=config,
                config_id=config_id,
                size_bytes=estimated_size,
                created_at=time.time(),
                last_accessed=time.time(),
                metadata=metadata or {},
            )

            self._cache[config_id] = entry
            self._estimated_memory_usage += estimated_size

            if self._stats:
                self._stats.record_set()

        self.logger.debug(f"Cached config {config_id} (size: {estimated_size} bytes)")

    async def invalidate(self, config_id: str) -> bool:
        """
        Remove a specific configuration from the cache.

        Args:
            config_id: Configuration ID to remove

        Returns:
            True if configuration was found and removed
        """
        with self._cache_lock:
            if config_id in self._cache:
                self._remove_entry(config_id)
                if self._stats:
                    self._stats.record_eviction()
                self.logger.debug(f"Invalidated config {config_id}")
                return True
            return False

    async def clear(self) -> int:
        """
        Clear all cached configurations.

        Returns:
            Number of configurations that were cleared
        """
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            self._estimated_memory_usage = 0
            if self._stats:
                self._stats.record_clear(count)

        self.logger.info(f"Cleared {count} cached configurations")
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._cache_lock:
            current_size = len(self._cache)
            memory_usage_mb = self._estimated_memory_usage / (1024 * 1024)

            stats = {
                "current_size": current_size,
                "max_size": self.max_size,
                "memory_usage_mb": round(memory_usage_mb, 2),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": round(
                    memory_usage_mb / (self.max_memory_bytes / (1024 * 1024)) * 100, 1
                ),
                "ttl_seconds": self.ttl_seconds,
            }

            if self._stats:
                stats.update(self._stats.to_dict())

            return stats

    def get_cache_keys(self) -> set[str]:
        """Get all current cache keys."""
        with self._cache_lock:
            return set(self._cache.keys())

    def _remove_entry(self, config_id: str) -> None:
        """Remove an entry from the cache (called with lock held)."""
        entry = self._cache.pop(config_id, None)
        if entry:
            self._estimated_memory_usage -= entry.size_bytes

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if not self._cache:
            return

        oldest_key = next(iter(self._cache))
        self._remove_entry(oldest_key)

        if self._stats:
            self._stats.record_eviction()

        self.logger.debug(f"Evicted oldest config {oldest_key}")

    def _is_expired(self, entry: "CacheEntry") -> bool:
        """Check if a cache entry has expired."""
        return (time.time() - entry.created_at) > self.ttl_seconds

    def _estimate_config_size(self, config: ExperimentConfig) -> int:
        """
        Estimate the memory size of a configuration object.

        Args:
            config: Configuration to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            # Convert to JSON string and estimate size
            config_dict = config.model_dump()
            config_str = str(config_dict)

            # Rough estimation: string length * 2 (Unicode) + object overhead
            base_size = len(config_str) * 2 + 1024  # 1KB overhead

            # Add extra for nested objects
            model_count = len(config.models) if config.models else 0
            dataset_count = len(config.datasets) if config.datasets else 0
            extra_size = (model_count + dataset_count) * 512

            return base_size + extra_size

        except Exception:
            # Fallback to a reasonable default
            return 8192  # 8KB default

    async def _cleanup_loop(self) -> None:
        """Background cleanup task for expired entries."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60,  # Check every minute
                )
                break  # Shutdown requested
            except TimeoutError:
                pass  # Continue cleanup

            # Perform cleanup
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        expired_keys = []

        with self._cache_lock:
            current_time = time.time()
            for config_id, entry in self._cache.items():
                if (current_time - entry.created_at) > self.ttl_seconds:
                    expired_keys.append(config_id)

        # Remove expired entries
        for config_id in expired_keys:
            await self.invalidate(config_id)

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class CacheEntry:
    """Represents a single cache entry with metadata."""

    def __init__(
        self,
        config: ExperimentConfig,
        config_id: str,
        size_bytes: int,
        created_at: float,
        last_accessed: float,
        metadata: dict[str, Any],
    ):
        self.config = config
        self.config_id = config_id
        self.size_bytes = size_bytes
        self.created_at = created_at
        self.last_accessed = last_accessed
        self.metadata = metadata


class CacheStats:
    """Cache statistics collector."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.clears = 0
        self._lock = threading.Lock()

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

    def record_set(self) -> None:
        """Record a cache set operation."""
        with self._lock:
            self.sets += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        with self._lock:
            self.evictions += 1

    def record_clear(self, count: int) -> None:
        """Record a cache clear operation."""
        with self._lock:
            self.clears += count

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "evictions": self.evictions,
                "total_requests": total_requests,
                "hit_rate_percent": round(hit_rate, 2),
                "clears": self.clears,
            }
