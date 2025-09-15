"""
Storage module for benchmark evaluation results.

This module provides SQLite-based storage and retrieval capabilities for
evaluation results with efficient querying and export functionality.
"""

from .results_storage import DatabaseMigrationError, ResultsStorage

__all__ = ["ResultsStorage", "DatabaseMigrationError"]
