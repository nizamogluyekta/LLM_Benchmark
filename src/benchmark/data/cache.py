"""
Efficient data caching system for processed datasets.

This module provides sophisticated caching capabilities for datasets with
file-based compression, in-memory storage, cache invalidation, and monitoring.
"""

import asyncio
import hashlib
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmark.core.logging import get_logger
from benchmark.data.models import Dataset


@dataclass
class CacheStats:
    """Cache statistics and monitoring data."""

    total_entries: int = 0
    memory_entries: int = 0
    file_entries: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    last_cleanup: datetime | None = None

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total_requests = self.hit_count + self.miss_count
        return self.hit_count / total_requests if total_requests > 0 else 0.0


@dataclass
class CacheEntry:
    """Individual cache entry metadata."""

    cache_key: str
    dataset_id: str
    config_hash: str
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    is_in_memory: bool = False
    file_path: Path | None = None


class DataCache:
    """
    Sophisticated caching system for datasets.

    Features:
    - File-based caching with compression
    - In-memory cache for frequently accessed data
    - Cache invalidation based on configuration changes
    - Automatic cache cleanup for old data
    - Cache statistics and monitoring
    - Support for partial dataset caching
    """

    def __init__(
        self,
        cache_dir: Path,
        max_memory_mb: int = 1000,
        max_disk_mb: int = 10000,
        compression_level: int = 6,
        memory_ttl_hours: int = 24,
        disk_ttl_days: int = 7,
    ):
        """
        Initialize the data cache.

        Args:
            cache_dir: Directory for file-based cache storage
            max_memory_mb: Maximum memory cache size in MB
            max_disk_mb: Maximum disk cache size in MB
            compression_level: Compression level (1-9, higher = better compression)
            memory_ttl_hours: Time-to-live for memory cache entries in hours
            disk_ttl_days: Time-to-live for disk cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.compression_level = compression_level
        self.memory_ttl = timedelta(hours=memory_ttl_hours)
        self.disk_ttl = timedelta(days=disk_ttl_days)

        # In-memory storage
        self.memory_cache: dict[str, Dataset] = {}
        self.memory_usage = 0

        # Cache metadata
        self.cache_entries: dict[str, CacheEntry] = {}

        # Statistics
        self.stats = CacheStats()

        # Thread safety
        self._lock = asyncio.Lock()

        # Logger
        self.logger = get_logger("data_cache")

        # Load existing cache metadata
        asyncio.create_task(self._load_cache_metadata())

    async def get_cached_dataset(self, dataset_id: str, config_hash: str) -> Dataset | None:
        """
        Retrieve a cached dataset.

        Args:
            dataset_id: Unique dataset identifier
            config_hash: Hash of the dataset configuration

        Returns:
            Cached dataset if found, None otherwise
        """
        cache_key = self._generate_cache_key(dataset_id, config_hash)

        async with self._lock:
            # Check if entry exists
            if cache_key not in self.cache_entries:
                self.stats.miss_count += 1
                return None

            entry = self.cache_entries[cache_key]

            # Check if memory cached
            if entry.is_in_memory and cache_key in self.memory_cache:
                dataset = self.memory_cache[cache_key]
                await self._update_access_stats(entry)
                self.stats.hit_count += 1
                self.logger.debug(f"Memory cache hit for dataset {dataset_id}")
                return dataset

            # Try to load from disk
            if entry.file_path and entry.file_path.exists():
                try:
                    dataset = await self._load_from_disk(entry.file_path)

                    # Promote to memory cache if there's space
                    if self._can_fit_in_memory(entry.size_bytes):
                        await self._add_to_memory_cache(cache_key, dataset, entry)

                    await self._update_access_stats(entry)
                    self.stats.hit_count += 1
                    self.logger.debug(f"Disk cache hit for dataset {dataset_id}")
                    return dataset

                except Exception as e:
                    self.logger.error(f"Failed to load dataset from cache: {e}")
                    # Remove corrupted entry
                    await self._remove_cache_entry(cache_key)

            self.stats.miss_count += 1
            return None

    async def cache_dataset(self, dataset_id: str, config_hash: str, dataset: Dataset) -> None:
        """
        Cache a dataset.

        Args:
            dataset_id: Unique dataset identifier
            config_hash: Hash of the dataset configuration
            dataset: Dataset to cache
        """
        cache_key = self._generate_cache_key(dataset_id, config_hash)

        async with self._lock:
            # Estimate dataset size
            dataset_size = await self._estimate_dataset_size(dataset)

            # Create cache entry
            entry = CacheEntry(
                cache_key=cache_key,
                dataset_id=dataset_id,
                config_hash=config_hash,
                size_bytes=dataset_size,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
            )

            # Try to add to memory cache first
            if self._can_fit_in_memory(dataset_size):
                await self._add_to_memory_cache(cache_key, dataset, entry)
                self.logger.debug(
                    f"Cached dataset {dataset_id} in memory ({dataset_size / 1024 / 1024:.1f} MB)"
                )

            # Always save to disk for persistence
            file_path = await self._save_to_disk(cache_key, dataset)
            entry.file_path = file_path

            # Store cache entry
            self.cache_entries[cache_key] = entry
            self.stats.total_entries = len(self.cache_entries)

            # Save metadata
            await self._save_cache_metadata()

            self.logger.info(f"Successfully cached dataset {dataset_id}")

    async def invalidate_cache(self, dataset_id: str) -> None:
        """
        Invalidate all cache entries for a dataset.

        Args:
            dataset_id: Dataset identifier to invalidate
        """
        async with self._lock:
            keys_to_remove = [
                key for key, entry in self.cache_entries.items() if entry.dataset_id == dataset_id
            ]

            for key in keys_to_remove:
                await self._remove_cache_entry(key)

            if keys_to_remove:
                self.logger.info(
                    f"Invalidated {len(keys_to_remove)} cache entries for dataset {dataset_id}"
                )
                await self._save_cache_metadata()

    async def cleanup_old_cache(self, max_age_days: int = 7) -> None:
        """
        Clean up old cache entries.

        Args:
            max_age_days: Maximum age of cache entries to keep
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        async with self._lock:
            keys_to_remove = [
                key
                for key, entry in self.cache_entries.items()
                if entry.last_accessed < cutoff_date
            ]

            for key in keys_to_remove:
                await self._remove_cache_entry(key)

            # Clean up orphaned files
            await self._cleanup_orphaned_files()

            self.stats.last_cleanup = datetime.now()

            if keys_to_remove:
                self.logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
                await self._save_cache_metadata()

    async def get_cache_stats(self) -> CacheStats:
        """Get current cache statistics."""
        async with self._lock:
            # Update current stats
            self.stats.memory_entries = len(self.memory_cache)
            self.stats.file_entries = len(
                [e for e in self.cache_entries.values() if e.file_path and e.file_path.exists()]
            )
            self.stats.memory_usage_mb = self.memory_usage / (1024 * 1024)
            self.stats.disk_usage_mb = await self._calculate_disk_usage()

            return self.stats

    def _generate_cache_key(self, dataset_id: str, config_hash: str) -> str:
        """Generate a unique cache key."""
        combined = f"{dataset_id}:{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()

    async def _compress_dataset(self, dataset: Dataset) -> bytes:
        """Compress dataset for storage."""
        try:
            # Serialize to JSON first for better compression
            json_data = dataset.model_dump_json().encode("utf-8")

            # Compress with zlib
            compressed = zlib.compress(json_data, self.compression_level)

            self.logger.debug(
                f"Compressed dataset from {len(json_data)} to {len(compressed)} bytes "
                f"({len(compressed) / len(json_data) * 100:.1f}% of original)"
            )

            return compressed

        except Exception as e:
            self.logger.error(f"Failed to compress dataset: {e}")
            raise

    async def _decompress_dataset(self, data: bytes) -> Dataset:
        """Decompress dataset from storage."""
        try:
            # Decompress
            json_data = zlib.decompress(data)

            # Deserialize from JSON
            dataset = Dataset.model_validate_json(json_data.decode("utf-8"))

            return dataset

        except Exception as e:
            self.logger.error(f"Failed to decompress dataset: {e}")
            raise

    async def _estimate_dataset_size(self, dataset: Dataset) -> int:
        """Estimate the size of a dataset in memory."""
        try:
            # Use pickle to get a rough estimate
            pickled = pickle.dumps(dataset.model_dump())
            return len(pickled)
        except Exception:
            # Fallback: estimate based on JSON serialization
            json_str = dataset.model_dump_json()
            return len(json_str.encode("utf-8"))

    def _can_fit_in_memory(self, size_bytes: int) -> bool:
        """Check if dataset can fit in memory cache."""
        return (self.memory_usage + size_bytes) <= self.max_memory_bytes

    async def _add_to_memory_cache(
        self, cache_key: str, dataset: Dataset, entry: CacheEntry
    ) -> None:
        """Add dataset to memory cache."""
        # Make space if needed
        await self._evict_memory_if_needed(entry.size_bytes)

        # Add to memory cache
        self.memory_cache[cache_key] = dataset
        self.memory_usage += entry.size_bytes
        entry.is_in_memory = True

    async def _evict_memory_if_needed(self, incoming_size: int) -> None:
        """Evict items from memory cache to make space."""
        if not self._can_fit_in_memory(incoming_size):
            # Sort entries by last accessed time (LRU eviction)
            memory_entries = [
                (key, entry) for key, entry in self.cache_entries.items() if entry.is_in_memory
            ]
            memory_entries.sort(key=lambda x: x[1].last_accessed)

            # Evict oldest entries until we have space
            for key, entry in memory_entries:
                if self._can_fit_in_memory(incoming_size):
                    break

                # Remove from memory
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    self.memory_usage -= entry.size_bytes
                    entry.is_in_memory = False
                    self.stats.eviction_count += 1

                    self.logger.debug(f"Evicted {entry.dataset_id} from memory cache")

    async def _save_to_disk(self, cache_key: str, dataset: Dataset) -> Path:
        """Save dataset to disk with compression."""
        file_path = self.cache_dir / f"{cache_key}.cache"

        try:
            # Compress dataset
            compressed_data = await self._compress_dataset(dataset)

            # Write to file
            with open(file_path, "wb") as f:
                f.write(compressed_data)

            return file_path

        except Exception as e:
            self.logger.error(f"Failed to save dataset to disk: {e}")
            raise

    async def _load_from_disk(self, file_path: Path) -> Dataset:
        """Load dataset from disk with decompression."""
        try:
            with open(file_path, "rb") as f:
                compressed_data = f.read()

            dataset = await self._decompress_dataset(compressed_data)
            return dataset

        except Exception as e:
            self.logger.error(f"Failed to load dataset from disk: {e}")
            raise

    async def _update_access_stats(self, entry: CacheEntry) -> None:
        """Update access statistics for a cache entry."""
        entry.last_accessed = datetime.now()
        entry.access_count += 1

    async def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry completely."""
        if cache_key not in self.cache_entries:
            return

        entry = self.cache_entries[cache_key]

        # Remove from memory cache
        if entry.is_in_memory and cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            self.memory_usage -= entry.size_bytes

        # Remove file
        if entry.file_path and entry.file_path.exists():
            try:
                entry.file_path.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to delete cache file {entry.file_path}: {e}")

        # Remove metadata
        del self.cache_entries[cache_key]
        self.stats.total_entries = len(self.cache_entries)

    async def _cleanup_orphaned_files(self) -> None:
        """Remove orphaned cache files."""
        if not self.cache_dir.exists():
            return

        # Get all cache files
        cache_files = set(self.cache_dir.glob("*.cache"))

        # Get expected files
        expected_files = {
            entry.file_path for entry in self.cache_entries.values() if entry.file_path
        }

        # Remove orphaned files
        orphaned_files = cache_files - expected_files
        for file_path in orphaned_files:
            try:
                file_path.unlink()
                self.logger.debug(f"Removed orphaned cache file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove orphaned file {file_path}: {e}")

    async def _calculate_disk_usage(self) -> float:
        """Calculate total disk usage in MB."""
        total_bytes = 0

        for entry in self.cache_entries.values():
            if entry.file_path and entry.file_path.exists():
                try:
                    total_bytes += entry.file_path.stat().st_size
                except Exception:
                    continue

        return total_bytes / (1024 * 1024)

    async def _save_cache_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"

        try:
            metadata = {
                "entries": {
                    key: {
                        "dataset_id": entry.dataset_id,
                        "config_hash": entry.config_hash,
                        "size_bytes": entry.size_bytes,
                        "created_at": entry.created_at.isoformat(),
                        "last_accessed": entry.last_accessed.isoformat(),
                        "access_count": entry.access_count,
                        "file_path": str(entry.file_path) if entry.file_path else None,
                    }
                    for key, entry in self.cache_entries.items()
                },
                "stats": {
                    "hit_count": self.stats.hit_count,
                    "miss_count": self.stats.miss_count,
                    "eviction_count": self.stats.eviction_count,
                },
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")

    async def _load_cache_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / "cache_metadata.json"

        if not metadata_file.exists():
            return

        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Restore cache entries
            for key, entry_data in metadata.get("entries", {}).items():
                entry = CacheEntry(
                    cache_key=key,
                    dataset_id=entry_data["dataset_id"],
                    config_hash=entry_data["config_hash"],
                    size_bytes=entry_data["size_bytes"],
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    access_count=entry_data["access_count"],
                    is_in_memory=False,
                    file_path=Path(entry_data["file_path"]) if entry_data["file_path"] else None,
                )
                self.cache_entries[key] = entry

            # Restore stats
            stats_data = metadata.get("stats", {})
            self.stats.hit_count = stats_data.get("hit_count", 0)
            self.stats.miss_count = stats_data.get("miss_count", 0)
            self.stats.eviction_count = stats_data.get("eviction_count", 0)
            self.stats.total_entries = len(self.cache_entries)

            self.logger.info(f"Loaded {len(self.cache_entries)} cache entries from metadata")

        except Exception as e:
            self.logger.warning(f"Failed to load cache metadata: {e}")


# Partial caching support
class PartialDataCache(DataCache):
    """Cache for partial datasets (e.g., individual splits or batches)."""

    async def cache_dataset_split(
        self,
        dataset_id: str,
        split_name: str,
        config_hash: str,
        samples: list[Any],
        metadata: dict[str, Any],
    ) -> None:
        """Cache a dataset split separately."""
        # Create a partial dataset for the split
        from benchmark.data.models import Dataset, DatasetInfo

        # Create minimal dataset info
        info = DatasetInfo(
            name=f"{dataset_id}_{split_name}",
            source="cached_split",
            total_samples=len(samples),
            attack_samples=len([s for s in samples if getattr(s, "label", "") == "ATTACK"]),
            benign_samples=len([s for s in samples if getattr(s, "label", "") == "BENIGN"]),
            attack_types=[],
            updated_at=None,
            description=None,
            size_bytes=None,
            format=None,
            metadata=metadata,
        )

        # Create dataset with just the split samples
        dataset = Dataset(info=info, samples=samples, splits=None, checksum=None)

        # Use a modified cache key for splits
        split_cache_key = f"{dataset_id}:{split_name}:{config_hash}"
        cache_key = hashlib.sha256(split_cache_key.encode()).hexdigest()

        await self.cache_dataset(dataset_id, cache_key, dataset)

    async def get_cached_dataset_split(
        self, dataset_id: str, split_name: str, config_hash: str
    ) -> list[Any] | None:
        """Retrieve a cached dataset split."""
        split_cache_key = f"{dataset_id}:{split_name}:{config_hash}"
        cache_key = hashlib.sha256(split_cache_key.encode()).hexdigest()

        dataset = await self.get_cached_dataset(dataset_id, cache_key)
        return dataset.samples if dataset else None


# Cache configuration
class CacheConfig(BaseModel):
    """Configuration for data cache."""

    cache_dir: str = Field(default="./cache", description="Cache directory path")
    max_memory_mb: int = Field(default=1000, ge=100, description="Maximum memory cache size in MB")
    max_disk_mb: int = Field(default=10000, ge=1000, description="Maximum disk cache size in MB")
    compression_level: int = Field(default=6, ge=1, le=9, description="Compression level (1-9)")
    memory_ttl_hours: int = Field(default=24, ge=1, description="Memory cache TTL in hours")
    disk_ttl_days: int = Field(default=7, ge=1, description="Disk cache TTL in days")
    enable_compression: bool = Field(default=True, description="Enable compression")
    auto_cleanup_enabled: bool = Field(default=True, description="Enable automatic cleanup")
    cleanup_interval_hours: int = Field(default=6, ge=1, description="Cleanup interval in hours")
