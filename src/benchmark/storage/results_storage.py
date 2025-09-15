"""
SQLite-based storage for evaluation results.

This module provides comprehensive storage and retrieval capabilities for
benchmark evaluation results, including experiments, models, datasets,
and detailed evaluation metrics with efficient querying support.
"""

import csv
import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from benchmark.interfaces.evaluation_interfaces import EvaluationResult, EvaluationSummary


class DatabaseMigrationError(Exception):
    """Exception raised when database migration fails."""

    pass


class ResultsStorage:
    """SQLite-based storage for evaluation results with efficient querying."""

    def __init__(self, db_path: str = "results/benchmark_results.db"):
        """
        Initialize the results storage.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_path = Path(__file__).parent / "database_schema.sql"

    async def initialize(self) -> None:
        """Initialize database schema and apply migrations."""
        await self._apply_migrations()
        await self._create_indexes()
        await self._validate_schema()

    async def _apply_migrations(self) -> None:
        """Apply database migrations from schema file."""
        if not self._schema_path.exists():
            raise DatabaseMigrationError(f"Schema file not found: {self._schema_path}")

        try:
            with open(self._schema_path) as f:
                schema_sql = f.read()

            async with aiosqlite.connect(self.db_path) as db:
                await db.executescript(schema_sql)
                await db.commit()
        except Exception as e:
            raise DatabaseMigrationError(f"Failed to apply database migrations: {e}") from e

    async def _create_indexes(self) -> None:
        """Create additional performance indexes."""
        additional_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_results_created_at ON evaluation_results(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at)",
        ]

        async with aiosqlite.connect(self.db_path) as db:
            for index_sql in additional_indexes:
                await db.execute(index_sql)
            await db.commit()

    async def _validate_schema(self) -> None:
        """Validate that the database schema is correctly applied."""
        required_tables = [
            "schema_version",
            "experiments",
            "datasets",
            "models",
            "evaluations",
            "evaluation_results",
            "predictions",
        ]

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in await cursor.fetchall()}

            missing_tables = set(required_tables) - existing_tables
            if missing_tables:
                raise DatabaseMigrationError(f"Missing required tables: {missing_tables}")

    async def store_evaluation_result(self, result: EvaluationResult) -> str:
        """
        Store evaluation result in database.

        Args:
            result: EvaluationResult object to store

        Returns:
            Evaluation ID for the stored result
        """
        evaluation_id = self._generate_evaluation_id(result)

        async with aiosqlite.connect(self.db_path) as db:
            try:
                # Insert evaluation record
                await db.execute(
                    """
                    INSERT OR REPLACE INTO evaluations (
                        id, experiment_id, model_id, dataset_id,
                        started_at, completed_at, status, execution_time_seconds,
                        success, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        evaluation_id,
                        result.experiment_id,
                        result.model_id,
                        result.dataset_id,
                        result.timestamp,
                        result.timestamp,
                        "completed" if result.success else "failed",
                        result.execution_time_seconds,
                        result.success,
                        json.dumps(result.metadata) if result.metadata else None,
                    ),
                )

                # Insert metric results
                for metric_name, value in result.metrics.items():
                    metric_type = self._determine_metric_type(metric_name, result.detailed_results)

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO evaluation_results (
                            evaluation_id, metric_type, metric_name, value, metadata, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            evaluation_id,
                            metric_type,
                            metric_name,
                            float(value),
                            json.dumps(result.detailed_results.get(metric_type, {})),
                            datetime.now().isoformat(),
                        ),
                    )

                await db.commit()
                return evaluation_id

            except Exception as e:
                await db.rollback()
                raise RuntimeError(f"Failed to store evaluation result: {e}") from e

    def _generate_evaluation_id(self, result: EvaluationResult) -> str:
        """Generate unique evaluation ID."""
        id_data = f"{result.experiment_id}_{result.model_id}_{result.dataset_id}_{result.timestamp}"
        return hashlib.md5(id_data.encode()).hexdigest()[:16]

    def _determine_metric_type(self, metric_name: str, detailed_results: dict[str, Any]) -> str:
        """Determine metric type from metric name and detailed results."""
        metric_name_lower = metric_name.lower()

        if any(
            keyword in metric_name_lower
            for keyword in ["accuracy", "precision", "recall", "f1", "auc", "matthews"]
        ):
            return "accuracy"
        elif any(
            keyword in metric_name_lower
            for keyword in ["time", "latency", "throughput", "tokens", "performance", "consistency"]
        ):
            return "performance"
        elif "false_positive" in metric_name_lower or "fpr" in metric_name_lower:
            return "false_positive_rate"
        elif any(
            keyword in metric_name_lower
            for keyword in ["explanation", "explainability", "bleu", "rouge"]
        ):
            return "explainability"
        else:
            return "other"

    async def store_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Store experiment information."""
        config_hash = self._compute_config_hash(config) if config else ""

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    id, name, description, config_hash, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    name,
                    description,
                    config_hash,
                    datetime.now().isoformat(),
                    json.dumps(config) if config else None,
                ),
            )
            await db.commit()

    async def store_model_info(
        self,
        model_id: str,
        name: str,
        model_type: str,
        version: str | None = None,
        parameters_count: int | None = None,
        config: dict[str, Any] | None = None,
        architecture: str | None = None,
    ) -> None:
        """Store model information."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO models (
                    id, name, type, version, parameters_count, created_at,
                    config, architecture
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_id,
                    name,
                    model_type,
                    version,
                    parameters_count,
                    datetime.now().isoformat(),
                    json.dumps(config) if config else None,
                    architecture,
                ),
            )
            await db.commit()

    async def store_dataset_info(
        self,
        dataset_id: str,
        name: str,
        source: str,
        version: str | None = None,
        samples_count: int | None = None,
        metadata: dict[str, Any] | None = None,
        file_path: str | None = None,
    ) -> None:
        """Store dataset information."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO datasets (
                    id, name, source, version, samples_count, created_at,
                    metadata, file_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    dataset_id,
                    name,
                    source,
                    version,
                    samples_count,
                    datetime.now().isoformat(),
                    json.dumps(metadata) if metadata else None,
                    file_path,
                ),
            )
            await db.commit()

    async def store_predictions(
        self, evaluation_id: str, predictions: list[dict[str, Any]]
    ) -> None:
        """Store individual predictions for detailed analysis."""
        async with aiosqlite.connect(self.db_path) as db:
            for i, pred in enumerate(predictions):
                await db.execute(
                    """
                    INSERT INTO predictions (
                        evaluation_id, sample_id, input_text, prediction,
                        confidence, explanation, ground_truth, processing_time_ms,
                        created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        evaluation_id,
                        pred.get("sample_id", f"sample_{i}"),
                        pred.get("input_text", ""),
                        pred.get("prediction", ""),
                        pred.get("confidence"),
                        pred.get("explanation"),
                        pred.get("ground_truth"),
                        pred.get("processing_time_ms"),
                        datetime.now().isoformat(),
                        json.dumps(
                            {
                                k: v
                                for k, v in pred.items()
                                if k
                                not in [
                                    "sample_id",
                                    "input_text",
                                    "prediction",
                                    "confidence",
                                    "explanation",
                                    "ground_truth",
                                    "processing_time_ms",
                                ]
                            }
                        ),
                    ),
                )
            await db.commit()

    def _compute_config_hash(self, config: dict[str, Any]) -> str:
        """Compute deterministic hash for configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    async def get_evaluation_results(
        self,
        experiment_id: str | None = None,
        model_id: str | None = None,
        dataset_id: str | None = None,
        metric_type: str | None = None,
        metric_name: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query evaluation results with optional filters."""
        query = """
            SELECT
                e.id as evaluation_id,
                e.experiment_id,
                e.model_id,
                e.dataset_id,
                e.started_at,
                e.completed_at,
                e.execution_time_seconds,
                e.success,
                er.metric_type,
                er.metric_name,
                er.value,
                er.metadata as metric_metadata,
                exp.name as experiment_name,
                m.name as model_name,
                m.type as model_type,
                d.name as dataset_name
            FROM evaluations e
            JOIN evaluation_results er ON e.id = er.evaluation_id
            JOIN experiments exp ON e.experiment_id = exp.id
            JOIN models m ON e.model_id = m.id
            JOIN datasets d ON e.dataset_id = d.id
            WHERE 1=1
        """
        params = []

        if experiment_id:
            query += " AND e.experiment_id = ?"
            params.append(experiment_id)
        if model_id:
            query += " AND e.model_id = ?"
            params.append(model_id)
        if dataset_id:
            query += " AND e.dataset_id = ?"
            params.append(dataset_id)
        if metric_type:
            query += " AND er.metric_type = ?"
            params.append(metric_type)
        if metric_name:
            query += " AND er.metric_name = ?"
            params.append(metric_name)

        query += " ORDER BY e.completed_at DESC LIMIT ? OFFSET ?"
        params.extend([str(limit), str(offset)])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                result_dict = dict(row)
                # Parse JSON metadata if present
                if result_dict.get("metric_metadata"):
                    try:
                        result_dict["metric_metadata"] = json.loads(result_dict["metric_metadata"])
                    except (json.JSONDecodeError, TypeError):
                        result_dict["metric_metadata"] = {}
                results.append(result_dict)

            return results

    async def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get comprehensive summary for an experiment."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get experiment info
            cursor = await db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            experiment = await cursor.fetchone()

            if not experiment:
                return {}

            # Get evaluation statistics
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) as total_evaluations,
                    COUNT(CASE WHEN success = 1 THEN 1 END) as successful_evaluations,
                    COUNT(CASE WHEN success = 0 THEN 1 END) as failed_evaluations,
                    AVG(execution_time_seconds) as avg_execution_time,
                    MIN(started_at) as first_evaluation,
                    MAX(completed_at) as last_evaluation
                FROM evaluations WHERE experiment_id = ?
            """,
                (experiment_id,),
            )
            stats = await cursor.fetchone()

            # Get best metrics per type
            cursor = await db.execute(
                """
                SELECT
                    er.metric_type,
                    er.metric_name,
                    MAX(er.value) as best_value,
                    COUNT(*) as measurement_count
                FROM evaluations e
                JOIN evaluation_results er ON e.id = er.evaluation_id
                WHERE e.experiment_id = ? AND e.success = 1
                GROUP BY er.metric_type, er.metric_name
                ORDER BY er.metric_type, er.metric_name
            """,
                (experiment_id,),
            )
            best_metrics = await cursor.fetchall()

            # Get unique models and datasets
            cursor = await db.execute(
                """
                SELECT
                    COUNT(DISTINCT model_id) as unique_models,
                    COUNT(DISTINCT dataset_id) as unique_datasets
                FROM evaluations WHERE experiment_id = ?
            """,
                (experiment_id,),
            )
            counts = await cursor.fetchone()

            return {
                "experiment": dict(experiment) if experiment else {},
                "statistics": dict(stats) if stats else {},
                "counts": dict(counts) if counts else {},
                "best_metrics": [dict(row) for row in best_metrics],
            }

    async def compare_models(
        self, model_ids: list[str], metric_name: str = "f1_score", dataset_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Compare models on a specific metric."""
        if not model_ids:
            return []

        placeholders = ",".join("?" * len(model_ids))
        query = f"""
            SELECT
                m.name as model_name,
                m.id as model_id,
                m.type as model_type,
                d.name as dataset_name,
                d.id as dataset_id,
                AVG(er.value) as avg_metric_value,
                MAX(er.value) as best_metric_value,
                MIN(er.value) as worst_metric_value,
                COUNT(*) as evaluation_count
            FROM models m
            JOIN evaluations e ON m.id = e.model_id
            JOIN datasets d ON e.dataset_id = d.id
            JOIN evaluation_results er ON e.id = er.evaluation_id
            WHERE m.id IN ({placeholders})
            AND er.metric_name = ?
            AND e.success = 1
        """
        params = list(model_ids) + [metric_name]

        if dataset_id:
            query += " AND d.id = ?"
            params.append(dataset_id)

        query += """
            GROUP BY m.id, d.id
            ORDER BY avg_metric_value DESC
        """

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_metric_trends(
        self,
        metric_name: str,
        model_id: str | None = None,
        dataset_id: str | None = None,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get metric trends over time."""
        query = f"""
            SELECT
                DATE(e.completed_at) as date,
                AVG(er.value) as avg_value,
                MIN(er.value) as min_value,
                MAX(er.value) as max_value,
                COUNT(*) as evaluation_count,
                m.name as model_name,
                d.name as dataset_name
            FROM evaluations e
            JOIN evaluation_results er ON e.id = er.evaluation_id
            JOIN models m ON e.model_id = m.id
            JOIN datasets d ON e.dataset_id = d.id
            WHERE er.metric_name = ?
            AND e.success = 1
            AND e.completed_at >= datetime('now', '-{days} days')
        """

        params = [metric_name]

        if model_id:
            query += " AND e.model_id = ?"
            params.append(model_id)
        if dataset_id:
            query += " AND e.dataset_id = ?"
            params.append(dataset_id)

        query += """
            GROUP BY DATE(e.completed_at), e.model_id, e.dataset_id
            ORDER BY date DESC
        """

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def export_to_csv(self, query_params: dict[str, Any] | None = None) -> str:
        """Export evaluation results to CSV format."""
        results = await self.get_evaluation_results(**(query_params or {}))

        if not results:
            return ""

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()

        for result in results:
            # Convert any dict values to JSON strings for CSV compatibility
            csv_row = {}
            for k, v in result.items():
                if isinstance(v, dict):
                    csv_row[k] = json.dumps(v)
                else:
                    csv_row[k] = v
            writer.writerow(csv_row)

        return output.getvalue()

    async def export_experiment_report(self, experiment_id: str) -> dict[str, Any]:
        """Export comprehensive experiment report."""
        summary = await self.get_experiment_summary(experiment_id)
        if not summary:
            return {}

        # Get detailed results
        results = await self.get_evaluation_results(experiment_id=experiment_id)

        # Group results by evaluation
        evaluations = {}
        for result in results:
            eval_id = result["evaluation_id"]
            if eval_id not in evaluations:
                evaluations[eval_id] = {
                    "evaluation_id": eval_id,
                    "model_name": result["model_name"],
                    "dataset_name": result["dataset_name"],
                    "completed_at": result["completed_at"],
                    "execution_time_seconds": result["execution_time_seconds"],
                    "metrics": {},
                }
            evaluations[eval_id]["metrics"][result["metric_name"]] = result["value"]

        return {
            "summary": summary,
            "evaluations": list(evaluations.values()),
            "export_timestamp": datetime.now().isoformat(),
        }

    async def get_evaluation_summary_stats(
        self, experiment_id: str | None = None
    ) -> EvaluationSummary:
        """Get evaluation summary statistics."""
        query = """
            SELECT
                COUNT(*) as total_evaluations,
                COUNT(CASE WHEN success = 1 THEN 1 END) as successful_evaluations,
                COUNT(CASE WHEN success = 0 THEN 1 END) as failed_evaluations,
                AVG(execution_time_seconds) as average_execution_time,
                MIN(started_at) as earliest_evaluation,
                MAX(completed_at) as latest_evaluation
            FROM evaluations
        """
        params = []

        if experiment_id:
            query += " WHERE experiment_id = ?"
            params.append(experiment_id)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            stats = await cursor.fetchone()

            # Get metric summaries
            metric_query = """
                SELECT
                    er.metric_name,
                    AVG(er.value) as avg_value,
                    MIN(er.value) as min_value,
                    MAX(er.value) as max_value,
                    COUNT(*) as count
                FROM evaluation_results er
                JOIN evaluations e ON er.evaluation_id = e.id
                WHERE e.success = 1
            """

            if experiment_id:
                metric_query += " AND e.experiment_id = ?"

            metric_query += " GROUP BY er.metric_name"

            cursor = await db.execute(metric_query, params)
            metrics = await cursor.fetchall()

            # Get models and datasets
            entity_query = """
                SELECT
                    DISTINCT m.name as model_name,
                    d.name as dataset_name
                FROM evaluations e
                JOIN models m ON e.model_id = m.id
                JOIN datasets d ON e.dataset_id = d.id
            """

            if experiment_id:
                entity_query += " WHERE e.experiment_id = ?"

            cursor = await db.execute(entity_query, params)
            entities = await cursor.fetchall()

            if not stats:
                return EvaluationSummary(
                    total_evaluations=0,
                    successful_evaluations=0,
                    failed_evaluations=0,
                    average_execution_time=0.0,
                    metric_summaries={},
                    time_range={"earliest": "", "latest": ""},
                    models_evaluated=[],
                    datasets_evaluated=[],
                )

            return EvaluationSummary(
                total_evaluations=stats["total_evaluations"] or 0,
                successful_evaluations=stats["successful_evaluations"] or 0,
                failed_evaluations=stats["failed_evaluations"] or 0,
                average_execution_time=stats["average_execution_time"] or 0.0,
                metric_summaries={
                    metric["metric_name"]: {
                        "avg": metric["avg_value"],
                        "min": metric["min_value"],
                        "max": metric["max_value"],
                        "count": metric["count"],
                    }
                    for metric in metrics
                },
                time_range={
                    "earliest": stats["earliest_evaluation"] or "",
                    "latest": stats["latest_evaluation"] or "",
                },
                models_evaluated=[entity["model_name"] for entity in entities],
                datasets_evaluated=list({entity["dataset_name"] for entity in entities}),
            )

    async def cleanup_old_evaluations(self, days: int = 90) -> int:
        """Remove old evaluation data to manage storage."""
        cutoff_date = datetime.now().replace(microsecond=0) - timedelta(days=days)

        async with aiosqlite.connect(self.db_path) as db:
            # Count evaluations to be deleted
            cursor = await db.execute(
                "SELECT COUNT(*) FROM evaluations WHERE completed_at < ?",
                (cutoff_date.isoformat(),),
            )
            result = await cursor.fetchone()
            count = result[0] if result else 0

            # Delete old evaluations (cascading deletes handle related records)
            await db.execute(
                "DELETE FROM evaluations WHERE completed_at < ?", (cutoff_date.isoformat(),)
            )
            await db.commit()

            return int(count)

    async def optimize_database(self) -> None:
        """Optimize database performance."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("VACUUM")
            await db.execute("ANALYZE")
            await db.commit()

    async def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        async with aiosqlite.connect(self.db_path) as db:
            stats = {}

            # Table row counts
            tables = [
                "experiments",
                "models",
                "datasets",
                "evaluations",
                "evaluation_results",
                "predictions",
            ]

            for table in tables:
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                result = await cursor.fetchone()
                stats[f"{table}_count"] = result[0] if result else 0

            # Database file size
            stats["database_size_bytes"] = self.db_path.stat().st_size

            # Schema version
            cursor = await db.execute("SELECT MAX(version) FROM schema_version")
            result = await cursor.fetchone()
            stats["schema_version"] = result[0] if result else 0

            return stats

    async def close(self) -> None:
        """Close database connections and cleanup resources."""
        # aiosqlite connections are automatically closed when context managers exit
        # This method is for future extension if needed
        pass
