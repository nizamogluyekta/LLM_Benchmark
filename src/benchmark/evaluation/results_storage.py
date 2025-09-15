"""
Robust storage system for evaluation results and metadata.

This module provides a comprehensive SQLite-based storage system for
evaluation results with querying, export, and data integrity features.
"""

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .result_models import EvaluationResult, ModelPerformanceHistory, ResultsQuery


class StorageError(Exception):
    """Base exception for storage-related errors."""

    pass


class DataIntegrityError(StorageError):
    """Exception raised when data integrity validation fails."""

    pass


class ResultsStorage:
    """
    Comprehensive results storage system with SQLite backend.

    Provides storage, querying, and export functionality for evaluation
    results with data integrity validation and performance optimization.
    """

    def __init__(self, storage_path: str = "~/.benchmark_cache/results"):
        """
        Initialize results storage.

        Args:
            storage_path: Directory path for storage files
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "results.db"

        # Initialize database
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # Create main results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    evaluation_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processing_time REAL NOT NULL,
                    model_version TEXT,
                    dataset_version TEXT,
                    experiment_name TEXT,
                    notes TEXT,
                    error_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS result_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluation_results (evaluation_id) ON DELETE CASCADE,
                    UNIQUE(evaluation_id, metric_name)
                )
            """)

            # Create configuration table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS result_configurations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    config_key TEXT NOT NULL,
                    config_value TEXT NOT NULL,
                    config_type TEXT NOT NULL DEFAULT 'string',
                    FOREIGN KEY (evaluation_id) REFERENCES evaluation_results (evaluation_id) ON DELETE CASCADE,
                    UNIQUE(evaluation_id, config_key)
                )
            """)

            # Create raw responses table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    response_index INTEGER NOT NULL,
                    response_data TEXT NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluation_results (evaluation_id) ON DELETE CASCADE,
                    UNIQUE(evaluation_id, response_index)
                )
            """)

            # Create tags table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS result_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluation_results (evaluation_id) ON DELETE CASCADE,
                    UNIQUE(evaluation_id, tag)
                )
            """)

            # Create performance indexes
            self._create_indexes(conn)

            conn.commit()

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance optimization."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_model_name ON evaluation_results(model_name)",
            "CREATE INDEX IF NOT EXISTS idx_task_type ON evaluation_results(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_dataset_name ON evaluation_results(dataset_name)",
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluation_results(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_experiment_name ON evaluation_results(experiment_name)",
            "CREATE INDEX IF NOT EXISTS idx_success_rate ON evaluation_results(success_rate)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON result_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_value ON result_metrics(metric_value)",
            "CREATE INDEX IF NOT EXISTS idx_tags ON result_tags(tag)",
        ]

        for index_sql in indexes:
            conn.execute(index_sql)

    def store_evaluation_result(self, result: EvaluationResult) -> str:
        """
        Store complete evaluation result with metadata.

        Args:
            result: EvaluationResult object to store

        Returns:
            evaluation_id of stored result

        Raises:
            DataIntegrityError: If data validation fails
            StorageError: If storage operation fails
        """
        try:
            # Validate result data
            result._validate_data()

            with sqlite3.connect(self.db_path) as conn:
                # Check if result already exists
                cursor = conn.execute(
                    "SELECT evaluation_id FROM evaluation_results WHERE evaluation_id = ?",
                    (result.evaluation_id,),
                )
                exists = cursor.fetchone() is not None

                if exists:
                    # Update existing result
                    self._update_evaluation_result(conn, result)
                else:
                    # Insert new result
                    self._insert_evaluation_result(conn, result)

                conn.commit()
                return result.evaluation_id

        except Exception as e:
            raise StorageError(f"Failed to store evaluation result: {e}") from e

    def _insert_evaluation_result(self, conn: sqlite3.Connection, result: EvaluationResult) -> None:
        """Insert new evaluation result into database."""
        # Insert main result record
        conn.execute(
            """
            INSERT INTO evaluation_results (
                evaluation_id, model_name, task_type, dataset_name, timestamp,
                processing_time, model_version, dataset_version, experiment_name,
                notes, error_count, success_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                result.evaluation_id,
                result.model_name,
                result.task_type,
                result.dataset_name,
                result.timestamp.isoformat(),
                result.processing_time,
                result.model_version,
                result.dataset_version,
                result.experiment_name,
                result.notes,
                result.error_count,
                result.success_rate,
            ),
        )

        # Insert related data
        self._insert_related_data(conn, result)

    def _insert_related_data(self, conn: sqlite3.Connection, result: EvaluationResult) -> None:
        """Insert related data (metrics, config, responses, tags) for evaluation result."""
        # Insert metrics
        for metric_name, metric_value in result.metrics.items():
            conn.execute(
                """
                INSERT INTO result_metrics (evaluation_id, metric_name, metric_value)
                VALUES (?, ?, ?)
            """,
                (result.evaluation_id, metric_name, float(metric_value)),
            )

        # Insert configuration
        for config_key, config_value in result.configuration.items():
            config_type = type(config_value).__name__
            config_value_str = (
                json.dumps(config_value)
                if isinstance(config_value, dict | list)
                else str(config_value)
            )

            conn.execute(
                """
                INSERT INTO result_configurations (evaluation_id, config_key, config_value, config_type)
                VALUES (?, ?, ?, ?)
            """,
                (result.evaluation_id, config_key, config_value_str, config_type),
            )

        # Insert raw responses
        for i, response in enumerate(result.raw_responses):
            conn.execute(
                """
                INSERT INTO raw_responses (evaluation_id, response_index, response_data)
                VALUES (?, ?, ?)
            """,
                (result.evaluation_id, i, json.dumps(response)),
            )

        # Insert tags
        for tag in result.tags:
            conn.execute(
                """
                INSERT INTO result_tags (evaluation_id, tag)
                VALUES (?, ?)
            """,
                (result.evaluation_id, tag),
            )

    def _update_evaluation_result(self, conn: sqlite3.Connection, result: EvaluationResult) -> None:
        """Update existing evaluation result in database."""
        # Update main record
        conn.execute(
            """
            UPDATE evaluation_results SET
                model_name = ?, task_type = ?, dataset_name = ?, timestamp = ?,
                processing_time = ?, model_version = ?, dataset_version = ?,
                experiment_name = ?, notes = ?, error_count = ?, success_rate = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE evaluation_id = ?
        """,
            (
                result.model_name,
                result.task_type,
                result.dataset_name,
                result.timestamp.isoformat(),
                result.processing_time,
                result.model_version,
                result.dataset_version,
                result.experiment_name,
                result.notes,
                result.error_count,
                result.success_rate,
                result.evaluation_id,
            ),
        )

        # Delete and re-insert related data
        for table in ["result_metrics", "result_configurations", "raw_responses", "result_tags"]:
            conn.execute(f"DELETE FROM {table} WHERE evaluation_id = ?", (result.evaluation_id,))

        # Re-insert all related data (without main record)
        self._insert_related_data(conn, result)

    def query_results(
        self, filters: dict[str, Any] | ResultsQuery | None = None
    ) -> list[EvaluationResult]:
        """
        Query stored results with filtering.

        Args:
            filters: Dictionary of filter criteria or ResultsQuery object

        Returns:
            List of matching EvaluationResult objects
        """
        if filters is None:
            filters = {}

        # Convert dict to ResultsQuery if needed
        query = ResultsQuery(**filters) if isinstance(filters, dict) else filters

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Build SQL query
                sql, params = self._build_query_sql(query)

                # Execute query
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

                # Convert rows to EvaluationResult objects
                results = []
                for row in rows:
                    result = self._row_to_evaluation_result(conn, row)
                    results.append(result)

                return results

        except Exception as e:
            raise StorageError(f"Failed to query results: {e}") from e

    def _build_query_sql(self, query: ResultsQuery) -> tuple[str, list[Any]]:
        """Build SQL query from ResultsQuery parameters."""
        sql = "SELECT * FROM evaluation_results WHERE 1=1"
        params = []

        # Add filters
        if query.evaluation_id:
            sql += " AND evaluation_id = ?"
            params.append(query.evaluation_id)

        if query.model_name:
            sql += " AND model_name = ?"
            params.append(query.model_name)

        if query.task_type:
            sql += " AND task_type = ?"
            params.append(query.task_type)

        if query.dataset_name:
            sql += " AND dataset_name = ?"
            params.append(query.dataset_name)

        if query.experiment_name:
            sql += " AND experiment_name = ?"
            params.append(query.experiment_name)

        if query.start_date:
            sql += " AND timestamp >= ?"
            params.append(query.start_date.isoformat())

        if query.end_date:
            sql += " AND timestamp <= ?"
            params.append(query.end_date.isoformat())

        if query.min_success_rate is not None:
            sql += " AND success_rate >= ?"
            params.append(str(query.min_success_rate))

        # Add tag filters
        if query.tags:
            tag_placeholders = ",".join("?" * len(query.tags))
            sql += f" AND evaluation_id IN (SELECT evaluation_id FROM result_tags WHERE tag IN ({tag_placeholders}))"
            params.extend(query.tags)

        # Add metric filters
        for metric_name, (min_val, max_val) in query.metric_filters.items():
            sql += " AND evaluation_id IN (SELECT evaluation_id FROM result_metrics WHERE metric_name = ? AND metric_value BETWEEN ? AND ?)"
            params.extend([metric_name, str(min_val), str(max_val)])

        # Add sorting
        if query.sort_by == "timestamp":
            sql += f" ORDER BY timestamp {query.sort_order.upper()}"
        elif query.sort_by in ["model_name", "success_rate", "processing_time"]:
            sql += f" ORDER BY {query.sort_by} {query.sort_order.upper()}"
        elif query.sort_by.startswith("metric:"):
            metric_name = query.sort_by[7:]  # Remove "metric:" prefix
            sql += f" ORDER BY (SELECT metric_value FROM result_metrics WHERE evaluation_id = evaluation_results.evaluation_id AND metric_name = ?) {query.sort_order.upper()}"
            params.append(metric_name)

        # Add pagination
        if query.limit:
            sql += " LIMIT ?"
            params.append(str(query.limit))

        if query.offset:
            sql += " OFFSET ?"
            params.append(str(query.offset))

        return sql, params

    def _row_to_evaluation_result(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> EvaluationResult:
        """Convert database row to EvaluationResult object."""
        evaluation_id = row["evaluation_id"]

        # Get metrics
        metrics_cursor = conn.execute(
            "SELECT metric_name, metric_value FROM result_metrics WHERE evaluation_id = ?",
            (evaluation_id,),
        )
        metrics = {row[0]: row[1] for row in metrics_cursor.fetchall()}

        # Get configuration
        config_cursor = conn.execute(
            "SELECT config_key, config_value, config_type FROM result_configurations WHERE evaluation_id = ?",
            (evaluation_id,),
        )
        configuration = {}
        for config_row in config_cursor.fetchall():
            key, value_str, value_type = config_row
            if value_type in ["dict", "list"]:
                configuration[key] = json.loads(value_str)
            elif value_type == "int":
                configuration[key] = int(value_str)
            elif value_type == "float":
                configuration[key] = float(value_str)
            elif value_type == "bool":
                configuration[key] = value_str.lower() == "true"
            else:
                configuration[key] = value_str

        # Get raw responses
        responses_cursor = conn.execute(
            "SELECT response_data FROM raw_responses WHERE evaluation_id = ? ORDER BY response_index",
            (evaluation_id,),
        )
        raw_responses = [json.loads(row[0]) for row in responses_cursor.fetchall()]

        # Get tags
        tags_cursor = conn.execute(
            "SELECT tag FROM result_tags WHERE evaluation_id = ?", (evaluation_id,)
        )
        tags = [row[0] for row in tags_cursor.fetchall()]

        return EvaluationResult(
            evaluation_id=evaluation_id,
            model_name=row["model_name"],
            task_type=row["task_type"],
            dataset_name=row["dataset_name"],
            metrics=metrics,
            timestamp=datetime.fromisoformat(row["timestamp"]),
            configuration=configuration,
            raw_responses=raw_responses,
            processing_time=row["processing_time"],
            model_version=row["model_version"],
            dataset_version=row["dataset_version"],
            experiment_name=row["experiment_name"],
            tags=tags,
            notes=row["notes"],
            error_count=row["error_count"] or 0,
            success_rate=row["success_rate"] or 1.0,
        )

    def get_model_performance_history(self, model_name: str) -> ModelPerformanceHistory:
        """
        Get performance history for specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelPerformanceHistory object with all evaluations for the model
        """
        results = self.query_results({"model_name": model_name})
        history = ModelPerformanceHistory(model_name=model_name)

        for result in results:
            history.add_evaluation(result)

        return history

    def export_results(
        self,
        format: str = "json",
        filters: dict[str, Any] | None = None,
        output_path: str | None = None,
    ) -> str:
        """
        Export results in specified format.

        Args:
            format: Export format ("json" or "csv")
            filters: Query filters to apply
            output_path: Optional output file path

        Returns:
            Exported data as string or path to output file
        """
        results = self.query_results(filters)

        if format.lower() == "json":
            exported_data = self._export_json(results)
        elif format.lower() == "csv":
            exported_data = self._export_csv(results)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(exported_data)
            return str(output_file)

        return exported_data

    def _export_json(self, results: list[EvaluationResult]) -> str:
        """Export results as JSON."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "result_count": len(results),
            "results": [result.to_dict() for result in results],
        }
        return json.dumps(export_data, indent=2)

    def _export_csv(self, results: list[EvaluationResult]) -> str:
        """Export results as CSV."""
        if not results:
            return ""

        # Collect all unique metric names
        all_metrics: set[str] = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        # Collect all unique configuration keys
        all_config_keys: set[str] = set()
        for result in results:
            all_config_keys.update(result.configuration.keys())

        # Define CSV columns
        base_columns = [
            "evaluation_id",
            "model_name",
            "task_type",
            "dataset_name",
            "timestamp",
            "processing_time",
            "model_version",
            "dataset_version",
            "experiment_name",
            "tags",
            "notes",
            "error_count",
            "success_rate",
        ]

        metric_columns = [f"metric_{metric}" for metric in sorted(all_metrics)]
        config_columns = [f"config_{key}" for key in sorted(all_config_keys)]

        all_columns = base_columns + metric_columns + config_columns

        # Create CSV content
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(all_columns)

        # Write data rows
        for result in results:
            row = [
                result.evaluation_id,
                result.model_name,
                result.task_type,
                result.dataset_name,
                result.timestamp.isoformat(),
                result.processing_time,
                result.model_version or "",
                result.dataset_version or "",
                result.experiment_name or "",
                ",".join(result.tags),
                result.notes or "",
                result.error_count,
                result.success_rate,
            ]

            # Add metric values
            for metric in sorted(all_metrics):
                row.append(result.metrics.get(metric, ""))

            # Add configuration values
            for key in sorted(all_config_keys):
                value = result.configuration.get(key, "")
                if isinstance(value, dict | list):
                    value = json.dumps(value)
                row.append(str(value))

            writer.writerow(row)

        return output.getvalue()

    def get_evaluation_by_id(self, evaluation_id: str) -> EvaluationResult | None:
        """Get specific evaluation result by ID."""
        results = self.query_results({"evaluation_id": evaluation_id})
        return results[0] if results else None

    def delete_evaluation(self, evaluation_id: str) -> bool:
        """
        Delete an evaluation result and all related data.

        Args:
            evaluation_id: ID of evaluation to delete

        Returns:
            True if deletion was successful, False if evaluation not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM evaluation_results WHERE evaluation_id = ?", (evaluation_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            raise StorageError(f"Failed to delete evaluation: {e}") from e

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics and metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get table counts
                stats = {}

                cursor = conn.execute("SELECT COUNT(*) FROM evaluation_results")
                stats["total_evaluations"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(DISTINCT model_name) FROM evaluation_results")
                stats["unique_models"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(DISTINCT task_type) FROM evaluation_results")
                stats["unique_task_types"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(DISTINCT dataset_name) FROM evaluation_results")
                stats["unique_datasets"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM result_metrics")
                stats["total_metrics"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM raw_responses")
                stats["total_responses"] = cursor.fetchone()[0]

                # Get date range
                cursor = conn.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM evaluation_results"
                )
                min_date, max_date = cursor.fetchone()
                stats["date_range"] = {"earliest": min_date, "latest": max_date}

                # Get database file size
                stats["database_size_bytes"] = (
                    self.db_path.stat().st_size if self.db_path.exists() else 0
                )

                return stats

        except Exception as e:
            raise StorageError(f"Failed to get storage stats: {e}") from e

    def validate_data_integrity(self) -> dict[str, Any]:
        """
        Validate data integrity across all stored results.

        Returns:
            Dictionary with validation results and any issues found
        """
        issues = []
        stats = {"total_checked": 0, "issues_found": 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check for orphaned records
                cursor = conn.execute("""
                    SELECT evaluation_id FROM result_metrics
                    WHERE evaluation_id NOT IN (SELECT evaluation_id FROM evaluation_results)
                """)
                orphaned_metrics = cursor.fetchall()
                if orphaned_metrics:
                    issues.append(f"Found {len(orphaned_metrics)} orphaned metric records")

                cursor = conn.execute("""
                    SELECT evaluation_id FROM result_configurations
                    WHERE evaluation_id NOT IN (SELECT evaluation_id FROM evaluation_results)
                """)
                orphaned_configs = cursor.fetchall()
                if orphaned_configs:
                    issues.append(f"Found {len(orphaned_configs)} orphaned configuration records")

                cursor = conn.execute("""
                    SELECT evaluation_id FROM raw_responses
                    WHERE evaluation_id NOT IN (SELECT evaluation_id FROM evaluation_results)
                """)
                orphaned_responses = cursor.fetchall()
                if orphaned_responses:
                    issues.append(f"Found {len(orphaned_responses)} orphaned response records")

                cursor = conn.execute("""
                    SELECT evaluation_id FROM result_tags
                    WHERE evaluation_id NOT IN (SELECT evaluation_id FROM evaluation_results)
                """)
                orphaned_tags = cursor.fetchall()
                if orphaned_tags:
                    issues.append(f"Found {len(orphaned_tags)} orphaned tag records")

                # Check for invalid data ranges
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_results WHERE success_rate < 0 OR success_rate > 1"
                )
                invalid_success_rates = cursor.fetchone()[0]
                if invalid_success_rates:
                    issues.append(
                        f"Found {invalid_success_rates} records with invalid success rates"
                    )

                cursor = conn.execute(
                    "SELECT COUNT(*) FROM evaluation_results WHERE processing_time < 0"
                )
                invalid_processing_times = cursor.fetchone()[0]
                if invalid_processing_times:
                    issues.append(
                        f"Found {invalid_processing_times} records with negative processing times"
                    )

                cursor = conn.execute("SELECT COUNT(*) FROM evaluation_results")
                stats["total_checked"] = cursor.fetchone()[0]
                stats["issues_found"] = len(issues)

                return {
                    "validation_timestamp": datetime.now().isoformat(),
                    "statistics": stats,
                    "issues": issues,
                    "integrity_ok": len(issues) == 0,
                }

        except Exception as e:
            raise StorageError(f"Failed to validate data integrity: {e}") from e
