"""
Model caching system for efficient model state persistence and retrieval.

This module provides sophisticated caching capabilities for LLM models, including
state serialization, compression, and intelligent cache management optimized
for Apple Silicon M4 Pro hardware and unified memory architecture.
"""

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from benchmark.core.config import ModelConfig


class CacheStatus(Enum):
    """Cache entry status."""

    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    CORRUPTED = "corrupted"


class CompressionLevel(Enum):
    """Compression levels for cached model states."""

    NONE = 0
    FAST = 1
    BALANCED = 6
    MAXIMUM = 9


@dataclass
class CacheEntry:
    """Information about a cached model state."""

    model_id: str
    config_hash: str
    file_path: Path
    creation_time: datetime
    last_accessed: datetime
    access_count: int
    file_size_bytes: int
    compressed: bool
    compression_level: CompressionLevel
    metadata: dict[str, Any]
    ttl_hours: float = 24.0


@dataclass
class CacheStatistics:
    """Cache performance and usage statistics."""

    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    eviction_count: int
    compression_ratio: float
    average_access_time_ms: float
    cache_efficiency_percent: float
    oldest_entry_age_hours: float
    newest_entry_age_hours: float


class ModelCache:
    """Advanced model caching system with compression and intelligent management."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_cache_size_gb: float = 8.0,
        default_ttl_hours: float = 24.0,
        cleanup_interval_minutes: int = 60,
    ):
        """
        Initialize model cache system.

        Args:
            cache_dir: Directory for cache storage
            max_cache_size_gb: Maximum cache size in GB
            default_ttl_hours: Default time-to-live for cache entries
            cleanup_interval_minutes: Automatic cleanup interval
        """
        self.cache_dir = cache_dir or Path.home() / ".benchmark_cache" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cache_size_bytes = int(max_cache_size_gb * 1024**3)
        self.default_ttl_hours = default_ttl_hours
        self.cleanup_interval_seconds = cleanup_interval_minutes * 60

        # Cache management
        self.cache_entries: dict[str, CacheEntry] = {}
        self.cache_index_file = self.cache_dir / "cache_index.json"

        # Statistics tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.access_times: list[float] = []

        self.logger = logging.getLogger(__name__)

        # Background cleanup task
        self._cleanup_task = None
        self._running = False

    async def initialize(self) -> None:
        """Initialize the cache system and load existing index."""
        await self._load_cache_index()
        await self._validate_cache_entries()

        self._running = True
        self._cleanup_task = asyncio.create_task(self._background_cleanup())

        self.logger.info(
            f"Model cache initialized: {len(self.cache_entries)} entries, "
            f"{await self._get_total_cache_size() / 1024**3:.2f}GB"
        )

    async def shutdown(self) -> None:
        """Shutdown the cache system."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        await self._save_cache_index()
        self.logger.info("Model cache shutdown complete")

    async def cache_model_state(
        self,
        model_id: str,
        config: ModelConfig,
        model_state: Any,
        metadata: dict[str, Any] | None = None,
        ttl_hours: float | None = None,
        compression: CompressionLevel = CompressionLevel.BALANCED,
    ) -> bool:
        """
        Cache model state with compression and metadata.

        Args:
            model_id: Unique model identifier
            config: Model configuration
            model_state: Model state object to cache
            metadata: Additional metadata to store
            ttl_hours: Time-to-live in hours
            compression: Compression level to use

        Returns:
            True if caching succeeded, False otherwise
        """
        try:
            start_time = time.time()

            # Generate unique cache key
            config_hash = self._generate_config_hash(config)
            cache_key = f"{model_id}_{config_hash}"

            # Prepare cache file path
            cache_file = self.cache_dir / f"{cache_key}.cache"

            # Serialize and optionally compress model state
            serialized_data = pickle.dumps(model_state)

            if compression != CompressionLevel.NONE:
                serialized_data = gzip.compress(serialized_data, compresslevel=compression.value)
                compressed = True
            else:
                compressed = False

            # Write to cache file
            cache_file.write_bytes(serialized_data)

            # Create cache entry
            now = datetime.now()
            cache_entry = CacheEntry(
                model_id=model_id,
                config_hash=config_hash,
                file_path=cache_file,
                creation_time=now,
                last_accessed=now,
                access_count=1,
                file_size_bytes=len(serialized_data),
                compressed=compressed,
                compression_level=compression,
                metadata=metadata or {},
                ttl_hours=ttl_hours or self.default_ttl_hours,
            )

            # Add to cache index
            self.cache_entries[cache_key] = cache_entry

            # Check if we need to evict entries
            await self._enforce_cache_limits()

            # Record access time
            access_time = (time.time() - start_time) * 1000
            self.access_times.append(access_time)

            # Keep only last 1000 access times for statistics
            if len(self.access_times) > 1000:
                self.access_times = self.access_times[-1000:]

            self.logger.info(
                f"Cached model {model_id}: {len(serialized_data) / 1024**2:.2f}MB "
                f"({'compressed' if compressed else 'uncompressed'})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to cache model {model_id}: {e}")
            return False

    async def load_cached_model(self, model_id: str, config: ModelConfig) -> Any | None:
        """
        Load model state from cache.

        Args:
            model_id: Model identifier
            config: Model configuration

        Returns:
            Cached model state or None if not found/invalid
        """
        try:
            start_time = time.time()

            # Generate cache key
            config_hash = self._generate_config_hash(config)
            cache_key = f"{model_id}_{config_hash}"

            # Check if entry exists
            if cache_key not in self.cache_entries:
                self.miss_count += 1
                return None

            cache_entry = self.cache_entries[cache_key]

            # Check cache status
            status = await self._get_cache_status(cache_entry)
            if status in [CacheStatus.EXPIRED, CacheStatus.CORRUPTED]:
                self.miss_count += 1
                await self._remove_cache_entry(cache_key)
                return None

            # Load and deserialize
            if not cache_entry.file_path.exists():
                self.miss_count += 1
                await self._remove_cache_entry(cache_key)
                return None

            cached_data = cache_entry.file_path.read_bytes()

            # Decompress if needed
            if cache_entry.compressed:
                cached_data = gzip.decompress(cached_data)

            model_state = pickle.loads(cached_data)

            # Update access statistics
            cache_entry.last_accessed = datetime.now()
            cache_entry.access_count += 1

            # Record access time
            access_time = (time.time() - start_time) * 1000
            self.access_times.append(access_time)

            if len(self.access_times) > 1000:
                self.access_times = self.access_times[-1000:]

            self.hit_count += 1

            self.logger.info(
                f"Loaded cached model {model_id}: "
                f"{cache_entry.file_size_bytes / 1024**2:.2f}MB "
                f"(hit #{cache_entry.access_count})"
            )

            return model_state

        except Exception as e:
            self.logger.error(f"Failed to load cached model {model_id}: {e}")
            self.miss_count += 1
            return None

    async def invalidate_cache_entry(self, model_id: str, config: ModelConfig) -> bool:
        """
        Invalidate specific cache entry.

        Args:
            model_id: Model identifier
            config: Model configuration

        Returns:
            True if entry was invalidated, False if not found
        """
        config_hash = self._generate_config_hash(config)
        cache_key = f"{model_id}_{config_hash}"

        if cache_key in self.cache_entries:
            await self._remove_cache_entry(cache_key)
            self.logger.info(f"Invalidated cache entry for {model_id}")
            return True

        return False

    async def clear_expired_entries(self) -> dict[str, int]:
        """
        Clear all expired cache entries.

        Returns:
            Statistics about cleared entries
        """
        cleared_count = 0
        freed_bytes = 0

        expired_keys = []
        for cache_key, entry in self.cache_entries.items():
            status = await self._get_cache_status(entry)
            if status in [CacheStatus.EXPIRED, CacheStatus.CORRUPTED]:
                expired_keys.append(cache_key)
                freed_bytes += entry.file_size_bytes

        for cache_key in expired_keys:
            await self._remove_cache_entry(cache_key)
            cleared_count += 1

        self.logger.info(
            f"Cleared {cleared_count} expired entries, freed {freed_bytes / 1024**2:.2f}MB"
        )

        return {"cleared_count": cleared_count, "freed_bytes": freed_bytes}

    async def optimize_cache(self) -> dict[str, Any]:
        """
        Optimize cache by recompressing entries and cleaning up fragmentation.

        Returns:
            Optimization statistics
        """
        optimization_stats = {
            "recompressed_count": 0,
            "space_saved_bytes": 0,
            "defragmented_count": 0,
            "optimization_time_seconds": 0,
        }

        start_time = time.time()

        try:
            # Recompress entries with suboptimal compression
            for _cache_key, entry in self.cache_entries.items():
                if (
                    entry.compression_level == CompressionLevel.FAST
                    and await self._recompress_entry(entry, CompressionLevel.BALANCED)
                ):
                    optimization_stats["recompressed_count"] += 1

            # Clear expired entries
            cleanup_result = await self.clear_expired_entries()
            optimization_stats["space_saved_bytes"] += cleanup_result["freed_bytes"]

            # Defragment cache directory
            await self._defragment_cache_directory()
            optimization_stats["defragmented_count"] = 1

            optimization_stats["optimization_time_seconds"] = time.time() - start_time

            self.logger.info(
                f"Cache optimization completed: "
                f"recompressed {optimization_stats['recompressed_count']} entries, "
                f"saved {optimization_stats['space_saved_bytes'] / 1024**2:.2f}MB"
            )

        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")
            optimization_stats["error"] = str(e)

        return optimization_stats

    async def get_cache_statistics(self) -> CacheStatistics:
        """Get comprehensive cache statistics."""
        try:
            total_size_bytes = await self._get_total_cache_size()
            total_requests = self.hit_count + self.miss_count

            # Calculate compression ratio
            uncompressed_size = 0
            compressed_size = 0
            for entry in self.cache_entries.values():
                if entry.compressed:
                    compressed_size += entry.file_size_bytes
                    # Estimate original size (rough approximation)
                    uncompressed_size += entry.file_size_bytes * 3
                else:
                    uncompressed_size += entry.file_size_bytes
                    compressed_size += entry.file_size_bytes

            compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0

            # Calculate age statistics
            now = datetime.now()
            ages = [
                (now - entry.creation_time).total_seconds() / 3600
                for entry in self.cache_entries.values()
            ]

            oldest_age = max(ages) if ages else 0.0
            newest_age = min(ages) if ages else 0.0

            # Cache efficiency
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0.0

            # Average access time
            avg_access_time = (
                sum(self.access_times) / len(self.access_times) if self.access_times else 0.0
            )

            return CacheStatistics(
                total_entries=len(self.cache_entries),
                total_size_bytes=total_size_bytes,
                hit_count=self.hit_count,
                miss_count=self.miss_count,
                eviction_count=self.eviction_count,
                compression_ratio=compression_ratio,
                average_access_time_ms=avg_access_time,
                cache_efficiency_percent=hit_rate,
                oldest_entry_age_hours=oldest_age,
                newest_entry_age_hours=newest_age,
            )

        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return CacheStatistics(0, 0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)

    def _generate_config_hash(self, config: ModelConfig) -> str:
        """Generate hash for model configuration."""
        config_dict = {
            "model_name": getattr(config, "model_name", ""),
            "type": getattr(config, "type", ""),
            "parameters": getattr(config, "parameters", {}),
            "context_length": getattr(config, "context_length", 0),
        }

        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    async def _get_cache_status(self, entry: CacheEntry) -> CacheStatus:
        """Determine cache entry status."""
        try:
            # Check if file exists
            if not entry.file_path.exists():
                return CacheStatus.CORRUPTED

            # Check file size consistency
            actual_size = entry.file_path.stat().st_size
            if actual_size != entry.file_size_bytes:
                return CacheStatus.CORRUPTED

            # Check TTL
            age_hours = (datetime.now() - entry.creation_time).total_seconds() / 3600
            if age_hours > entry.ttl_hours:
                return CacheStatus.EXPIRED

            # Check staleness (configurable threshold)
            stale_threshold_hours = entry.ttl_hours * 0.8
            if age_hours > stale_threshold_hours:
                return CacheStatus.STALE

            return CacheStatus.FRESH

        except Exception:
            return CacheStatus.CORRUPTED

    async def _enforce_cache_limits(self) -> None:
        """Enforce cache size limits by evicting entries."""
        total_size = await self._get_total_cache_size()

        if total_size <= self.max_cache_size_bytes:
            return

        # Sort entries by eviction priority (LRU + size consideration)
        entries_with_priority = []
        for cache_key, entry in self.cache_entries.items():
            priority = self._calculate_eviction_priority(entry)
            entries_with_priority.append((cache_key, entry, priority))

        # Sort by priority (higher = more likely to evict)
        entries_with_priority.sort(key=lambda x: x[2], reverse=True)

        # Evict entries until we're under the limit
        for cache_key, entry, _ in entries_with_priority:
            if total_size <= self.max_cache_size_bytes:
                break

            await self._remove_cache_entry(cache_key)
            total_size -= entry.file_size_bytes
            self.eviction_count += 1

    def _calculate_eviction_priority(self, entry: CacheEntry) -> float:
        """Calculate eviction priority (higher = more likely to evict)."""
        now = datetime.now()

        # Time since last access (higher = more likely to evict)
        hours_since_access = (now - entry.last_accessed).total_seconds() / 3600

        # File size consideration (larger files = slightly more likely to evict)
        size_factor = entry.file_size_bytes / (1024**3)  # GB

        # Access frequency (lower frequency = more likely to evict)
        hours_alive = (now - entry.creation_time).total_seconds() / 3600
        access_rate = entry.access_count / max(hours_alive, 1.0)

        # Calculate priority score
        priority = (hours_since_access * 2.0) + (size_factor * 0.1) - (access_rate * 1.0)

        return priority

    async def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove cache entry and associated file."""
        if cache_key in self.cache_entries:
            entry = self.cache_entries[cache_key]

            # Remove file
            try:
                if entry.file_path.exists():
                    entry.file_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {entry.file_path}: {e}")

            # Remove from index
            del self.cache_entries[cache_key]

    async def _get_total_cache_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(entry.file_size_bytes for entry in self.cache_entries.values())

    async def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        if not self.cache_index_file.exists():
            return

        try:
            with open(self.cache_index_file) as f:
                index_data = json.load(f)

            for cache_key, entry_data in index_data.items():
                entry = CacheEntry(
                    model_id=entry_data["model_id"],
                    config_hash=entry_data["config_hash"],
                    file_path=Path(entry_data["file_path"]),
                    creation_time=datetime.fromisoformat(entry_data["creation_time"]),
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    access_count=entry_data["access_count"],
                    file_size_bytes=entry_data["file_size_bytes"],
                    compressed=entry_data["compressed"],
                    compression_level=CompressionLevel(entry_data["compression_level"]),
                    metadata=entry_data["metadata"],
                    ttl_hours=entry_data["ttl_hours"],
                )
                self.cache_entries[cache_key] = entry

            self.logger.info(f"Loaded cache index: {len(self.cache_entries)} entries")

        except Exception as e:
            self.logger.error(f"Failed to load cache index: {e}")
            self.cache_entries = {}

    async def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        try:
            index_data = {}
            for cache_key, entry in self.cache_entries.items():
                index_data[cache_key] = {
                    "model_id": entry.model_id,
                    "config_hash": entry.config_hash,
                    "file_path": str(entry.file_path),
                    "creation_time": entry.creation_time.isoformat(),
                    "last_accessed": entry.last_accessed.isoformat(),
                    "access_count": entry.access_count,
                    "file_size_bytes": entry.file_size_bytes,
                    "compressed": entry.compressed,
                    "compression_level": entry.compression_level.value,
                    "metadata": entry.metadata,
                    "ttl_hours": entry.ttl_hours,
                }

            with open(self.cache_index_file, "w") as f:
                json.dump(index_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")

    async def _validate_cache_entries(self) -> None:
        """Validate existing cache entries and remove invalid ones."""
        invalid_keys = []

        for cache_key, entry in self.cache_entries.items():
            status = await self._get_cache_status(entry)
            if status == CacheStatus.CORRUPTED:
                invalid_keys.append(cache_key)

        for cache_key in invalid_keys:
            await self._remove_cache_entry(cache_key)
            self.logger.warning(f"Removed corrupted cache entry: {cache_key}")

    async def _recompress_entry(self, entry: CacheEntry, new_compression: CompressionLevel) -> bool:
        """Recompress cache entry with new compression level."""
        try:
            # Load current data
            if not entry.file_path.exists():
                return False

            current_data = entry.file_path.read_bytes()

            # Decompress if currently compressed
            decompressed_data = gzip.decompress(current_data) if entry.compressed else current_data

            # Recompress with new level
            if new_compression != CompressionLevel.NONE:
                new_data = gzip.compress(decompressed_data, compresslevel=new_compression.value)
                compressed = True
            else:
                new_data = decompressed_data
                compressed = False

            # Only update if we saved space
            if len(new_data) < len(current_data):
                entry.file_path.write_bytes(new_data)
                entry.file_size_bytes = len(new_data)
                entry.compressed = compressed
                entry.compression_level = new_compression
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to recompress entry: {e}")
            return False

    async def _defragment_cache_directory(self) -> None:
        """Clean up cache directory structure."""
        try:
            # Remove orphaned cache files (files without index entries)
            cache_files = set(self.cache_dir.glob("*.cache"))
            indexed_files = {entry.file_path for entry in self.cache_entries.values()}

            orphaned_files = cache_files - indexed_files
            for orphaned_file in orphaned_files:
                try:
                    orphaned_file.unlink()
                    self.logger.info(f"Removed orphaned cache file: {orphaned_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove orphaned file {orphaned_file}: {e}")

        except Exception as e:
            self.logger.error(f"Cache defragmentation failed: {e}")

    async def _background_cleanup(self) -> None:
        """Background task for cache maintenance."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)

                if not self._running:
                    break

                # Clear expired entries
                await self.clear_expired_entries()

                # Enforce cache limits
                await self._enforce_cache_limits()

                # Save index periodically
                await self._save_cache_index()

                self.logger.debug("Background cache cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in background cache cleanup: {e}")
