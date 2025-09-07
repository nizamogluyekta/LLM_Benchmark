"""
Lazy configuration loader for section-based loading and precompilation.

This module provides lazy loading capabilities for configuration files,
allowing sections to be loaded on-demand and common sections to be preloaded
for better performance.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from benchmark.core.logging import get_logger


class LazyConfigLoader:
    """
    Lazy loader for configuration sections with preloading support.

    Features:
    - Section-based loading for large configurations
    - Preloading of common sections
    - Configuration section caching
    - Diff tracking to avoid reprocessing
    - Async loading for better performance
    """

    def __init__(self, cache_size: int = 50):
        """
        Initialize the lazy config loader.

        Args:
            cache_size: Maximum number of section caches to maintain
        """
        self.cache_size = cache_size
        self.logger = get_logger("lazy_config_loader")

        # Section cache: {file_hash -> {section_name -> section_data}}
        self._section_cache: dict[str, dict[str, Any]] = {}

        # File modification tracking for change detection
        self._file_hashes: dict[str, str] = {}

        # Preloaded sections registry
        self._preloaded_sections: set[str] = set()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def load_section(self, config_path: str, section: str) -> Any:
        """
        Load a specific section from a configuration file.

        Args:
            config_path: Path to the configuration file
            section: Name of the section to load (e.g., 'models', 'datasets', 'evaluation')

        Returns:
            Dictionary containing the requested section data

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            KeyError: If section doesn't exist in configuration
            yaml.YAMLError: If YAML parsing fails
        """
        config_path_obj = Path(config_path).resolve()
        file_key = str(config_path_obj)

        async with self._lock:
            # Check if we need to reload due to file changes
            current_hash = await self._get_file_hash(config_path_obj)

            if (
                file_key not in self._section_cache
                or self._file_hashes.get(file_key) != current_hash
            ):
                # Load and cache all sections
                await self._load_all_sections(config_path_obj, file_key, current_hash)

            # Return requested section
            cached_sections = self._section_cache.get(file_key, {})
            if section not in cached_sections:
                raise KeyError(f"Section '{section}' not found in configuration {config_path_obj}")

            self.logger.debug(f"Loaded section '{section}' from {config_path_obj}")
            section_data = cached_sections[section]

            # Return a copy if it's a dict/list, otherwise return as-is for primitive types
            if isinstance(section_data, dict):
                return section_data.copy()
            elif isinstance(section_data, list):
                return list(section_data)
            else:
                return section_data

    async def preload_common_sections(self, config_paths: list[str]) -> None:
        """
        Preload common sections from multiple configuration files.

        This method identifies and preloads frequently used sections
        to improve performance for subsequent loads.

        Args:
            config_paths: List of configuration file paths to preload
        """
        self.logger.info(f"Preloading common sections from {len(config_paths)} configurations")

        # Define common sections that are frequently accessed
        common_sections = {"models", "datasets", "evaluation", "name", "description"}

        preload_tasks = []
        for config_path in config_paths:
            if Path(config_path).exists():
                for section in common_sections:
                    task = self._safe_preload_section(config_path, section)
                    preload_tasks.append(task)

        # Execute preloading tasks concurrently
        if preload_tasks:
            results = await asyncio.gather(*preload_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            self.logger.info(
                f"Preloaded {success_count}/{len(preload_tasks)} sections successfully"
            )

    async def get_config_outline(self, config_path: str) -> dict[str, Any]:
        """
        Get a lightweight outline of a configuration file.

        Returns only the structure and basic metadata without loading
        heavy sections like detailed model configurations.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary with configuration outline
        """
        config_path_obj = Path(config_path).resolve()

        try:
            async with self._lock:
                # Load minimal sections for outline
                outline_sections = ["name", "description", "output_dir"]
                outline: dict[str, Any] = {}

                for section in outline_sections:
                    try:
                        section_data = await self.load_section(str(config_path_obj), section)
                        outline[section] = section_data
                    except KeyError:
                        # Section doesn't exist, skip
                        pass

                # Add metadata about available sections
                file_key = str(config_path_obj)
                if file_key in self._section_cache:
                    outline["_available_sections"] = list(self._section_cache[file_key].keys())

                # Add counts for major sections
                try:
                    models = await self.load_section(str(config_path_obj), "models")
                    outline["_models_count"] = len(models) if isinstance(models, list) else 1
                except KeyError:
                    outline["_models_count"] = 0

                try:
                    datasets = await self.load_section(str(config_path_obj), "datasets")
                    outline["_datasets_count"] = len(datasets) if isinstance(datasets, list) else 1
                except KeyError:
                    outline["_datasets_count"] = 0

                return outline

        except Exception as e:
            self.logger.error(f"Failed to get config outline for {config_path}: {e}")
            raise

    async def is_config_modified(self, config_path: str) -> bool:
        """
        Check if a configuration file has been modified since last load.

        Args:
            config_path: Path to the configuration file

        Returns:
            True if file has been modified or not yet loaded
        """
        config_path_obj = Path(config_path).resolve()
        file_key = str(config_path_obj)

        if not config_path_obj.exists():
            return True  # File deleted, consider as modified

        current_hash = await self._get_file_hash(config_path_obj)
        return self._file_hashes.get(file_key) != current_hash

    async def clear_cache(self, config_path: str | None = None) -> None:
        """
        Clear cached sections for a specific file or all files.

        Args:
            config_path: Specific configuration path to clear, or None for all
        """
        async with self._lock:
            if config_path:
                file_key = str(Path(config_path).resolve())
                self._section_cache.pop(file_key, None)
                self._file_hashes.pop(file_key, None)
                self.logger.debug(f"Cleared cache for {config_path}")
            else:
                self._section_cache.clear()
                self._file_hashes.clear()
                self._preloaded_sections.clear()
                self.logger.info("Cleared all section caches")

    async def get_cache_info(self) -> dict[str, Any]:
        """
        Get information about the current cache state.

        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            total_sections = sum(len(sections) for sections in self._section_cache.values())

            return {
                "cached_files": len(self._section_cache),
                "total_sections": total_sections,
                "preloaded_sections": len(self._preloaded_sections),
                "cache_size_limit": self.cache_size,
                "cached_file_paths": list(self._section_cache.keys()),
            }

    async def _load_all_sections(self, config_path: Path, file_key: str, file_hash: str) -> None:
        """
        Load all sections from a configuration file.

        Args:
            config_path: Path to the configuration file
            file_key: Cache key for the file
            file_hash: Hash of the file content
        """
        try:
            # Read and parse YAML file
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if config_data is None:
                config_data = {}

            # Store sections in cache
            self._section_cache[file_key] = config_data
            self._file_hashes[file_key] = file_hash

            # Manage cache size
            await self._manage_cache_size()

            self.logger.debug(f"Loaded {len(config_data)} sections from {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load sections from {config_path}: {e}")
            raise

    async def _safe_preload_section(self, config_path: str, section: str) -> Any:
        """
        Safely preload a section without raising exceptions.

        Args:
            config_path: Configuration file path
            section: Section name to preload

        Returns:
            Section data or None if loading failed
        """
        try:
            section_data = await self.load_section(config_path, section)
            preload_key = f"{config_path}:{section}"
            self._preloaded_sections.add(preload_key)
            return section_data
        except Exception as e:
            self.logger.debug(f"Failed to preload section '{section}' from {config_path}: {e}")
            return None

    async def _get_file_hash(self, file_path: Path) -> str:
        """
        Get a hash of the file content for change detection.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash of the file content
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            # If we can't read the file, return a unique hash based on timestamp
            stat = file_path.stat()
            return hashlib.md5(str(stat.st_mtime).encode()).hexdigest()

    async def _manage_cache_size(self) -> None:
        """Manage cache size by removing oldest entries if necessary."""
        while len(self._section_cache) > self.cache_size:
            # Remove the oldest entry (first in the dict)
            oldest_key = next(iter(self._section_cache))
            del self._section_cache[oldest_key]
            self._file_hashes.pop(oldest_key, None)

            # Remove related preloaded sections
            to_remove = [
                key for key in self._preloaded_sections if key.startswith(oldest_key + ":")
            ]
            for key in to_remove:
                self._preloaded_sections.discard(key)

            self.logger.debug(f"Evicted cache entry for {oldest_key}")


class ConfigDiffTracker:
    """
    Track configuration changes to avoid unnecessary reprocessing.

    This class helps optimize performance by detecting what sections
    of a configuration have actually changed.
    """

    def __init__(self) -> None:
        """Initialize the diff tracker."""
        self.logger = get_logger("config_diff_tracker")
        self._previous_configs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def track_config(self, config_path: str, config_data: dict[str, Any]) -> None:
        """
        Start tracking a configuration for changes.

        Args:
            config_path: Path to the configuration file
            config_data: Current configuration data
        """
        config_key = str(Path(config_path).resolve())

        async with self._lock:
            self._previous_configs[config_key] = self._deep_copy_serializable(config_data)

    async def get_changed_sections(
        self, config_path: str, current_config: dict[str, Any]
    ) -> set[str]:
        """
        Get the sections that have changed since last tracking.

        Args:
            config_path: Path to the configuration file
            current_config: Current configuration data

        Returns:
            Set of section names that have changed
        """
        config_key = str(Path(config_path).resolve())

        async with self._lock:
            previous_config = self._previous_configs.get(config_key)
            if previous_config is None:
                # First time seeing this config, all sections are "changed"
                return set(current_config.keys())

            changed_sections = set()

            # Check for modified sections
            for section, current_value in current_config.items():
                previous_value = previous_config.get(section)
                if not self._deep_equal(current_value, previous_value):
                    changed_sections.add(section)

            # Check for removed sections
            for section in previous_config:
                if section not in current_config:
                    changed_sections.add(section)

            return changed_sections

    async def clear_tracking(self, config_path: str | None = None) -> None:
        """
        Clear tracking for a specific config or all configs.

        Args:
            config_path: Specific config path to clear, or None for all
        """
        async with self._lock:
            if config_path:
                config_key = str(Path(config_path).resolve())
                self._previous_configs.pop(config_key, None)
            else:
                self._previous_configs.clear()

    def _deep_copy_serializable(self, obj: Any) -> Any:
        """Create a deep copy of serializable objects."""
        try:
            return json.loads(json.dumps(obj, default=str, sort_keys=True))
        except (TypeError, ValueError):
            # Fallback for non-serializable objects
            return str(obj)

    def _deep_equal(self, obj1: Any, obj2: Any) -> bool:
        """Check if two objects are deeply equal."""
        try:
            # Convert to JSON strings for comparison
            str1 = json.dumps(obj1, default=str, sort_keys=True)
            str2 = json.dumps(obj2, default=str, sort_keys=True)
            return str1 == str2
        except (TypeError, ValueError):
            # Fallback to string comparison
            return str(obj1) == str(obj2)
