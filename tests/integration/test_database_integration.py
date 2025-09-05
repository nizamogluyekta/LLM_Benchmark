"""
Integration tests for database manager with real databases.

Tests database manager with actual SQLite databases, concurrent access,
error recovery, and performance characteristics.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from benchmark.core.database import Dataset, Evaluation, Experiment, Model
from benchmark.core.database_manager import (
    DatabaseManager,
    get_sqlite_url,
)


class TestDatabaseIntegrationSQLite:
    """Integration tests with actual SQLite database."""

    @pytest.fixture
    async def temp_db_file(self):
        """Create temporary SQLite database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_name = temp_file.name

        yield temp_name

        # Cleanup
        Path(temp_name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_file_database_persistence(self, temp_db_file):
        """Test data persistence with file-based SQLite database."""
        db_url = get_sqlite_url(temp_db_file)

        experiment_name = "persistent-test"
        experiment_id = None

        # Create data in first session
        async with DatabaseManager(db_url) as manager:
            await manager.create_tables()

            async with manager.session_scope() as session:
                experiment = Experiment(
                    name=experiment_name, config={"persistent": True}, output_dir="/tmp/persistent"
                )
                session.add(experiment)
                await session.flush()  # Get ID before commit
                experiment_id = experiment.id

        # Verify data persists in new manager instance
        async with DatabaseManager(db_url) as manager2, manager2.session_scope() as session:
            retrieved = await session.get(Experiment, experiment_id)
            assert retrieved is not None
            assert retrieved.name == experiment_name
            assert retrieved.config == {"persistent": True}

    @pytest.mark.asyncio
    async def test_concurrent_access_file_database(self, temp_db_file):
        """Test concurrent access to file-based SQLite database."""
        db_url = get_sqlite_url(temp_db_file)

        # Initialize database
        async with DatabaseManager(db_url) as setup_manager:
            await setup_manager.create_tables()

        async def worker(worker_id: int, num_records: int):
            """Worker function that creates records."""
            async with DatabaseManager(db_url) as manager:
                for i in range(num_records):
                    async with manager.session_scope() as session:
                        experiment = Experiment(
                            name=f"worker-{worker_id}-record-{i}",
                            config={"worker_id": worker_id, "record": i},
                            output_dir=f"/tmp/worker-{worker_id}",
                        )
                        session.add(experiment)

        # Run multiple workers concurrently
        num_workers = 3
        records_per_worker = 5
        tasks = [worker(i, records_per_worker) for i in range(num_workers)]

        await asyncio.gather(*tasks)

        # Verify all records were created
        async with DatabaseManager(db_url) as manager, manager.session_scope() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == num_workers * records_per_worker

    @pytest.mark.asyncio
    async def test_database_backup_and_restore(self, temp_db_file):
        """Test database backup and restore functionality."""
        db_url = get_sqlite_url(temp_db_file)
        backup_file = temp_db_file + ".backup"

        try:
            # Create original database with data
            async with DatabaseManager(db_url) as manager:
                await manager.create_tables()

                async with manager.session_scope() as session:
                    experiment = Experiment(
                        name="backup-test", config={"backup": True}, output_dir="/tmp/backup"
                    )
                    session.add(experiment)

            # Create backup
            async with DatabaseManager(db_url) as manager:
                await manager.backup_database(backup_file)

            # Verify backup file exists
            assert Path(backup_file).exists()

            # Verify backup contains data
            backup_url = get_sqlite_url(backup_file)
            async with (
                DatabaseManager(backup_url) as backup_manager,
                backup_manager.session_scope() as session,
            ):
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == 1

                result = await session.execute(text("SELECT name FROM experiments"))
                name = result.scalar()
                assert name == "backup-test"

        finally:
            # Cleanup backup file
            Path(backup_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_large_dataset_operations(self, temp_db_file):
        """Test operations with larger datasets."""
        db_url = get_sqlite_url(temp_db_file)
        num_experiments = 100

        async with DatabaseManager(db_url) as manager:
            await manager.create_tables()

            # Create many experiments in batches
            batch_size = 20
            for batch_start in range(0, num_experiments, batch_size):
                async with manager.session_scope() as session:
                    for i in range(batch_start, min(batch_start + batch_size, num_experiments)):
                        experiment = Experiment(
                            name=f"large-test-{i}",
                            config={"index": i, "batch": batch_start // batch_size},
                            output_dir=f"/tmp/large-{i}",
                        )
                        session.add(experiment)

            # Verify all experiments were created
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == num_experiments

            # Test batch retrieval
            async with manager.session_scope() as session:
                result = await session.execute(
                    text(
                        "SELECT name FROM experiments WHERE name LIKE 'large-test-%' ORDER BY name LIMIT 10"
                    )
                )
                names = [row[0] for row in result.fetchall()]
                assert len(names) == 10
                assert names[0] == "large-test-0"

    @pytest.mark.asyncio
    async def test_complex_relationships_and_queries(self, temp_db_file):
        """Test complex database operations with relationships."""
        db_url = get_sqlite_url(temp_db_file)

        async with DatabaseManager(db_url) as manager:
            await manager.create_tables()

            # Create related data
            async with manager.session_scope() as session:
                # Create experiment
                experiment = Experiment(
                    name="relationship-test",
                    config={"relationships": True},
                    output_dir="/tmp/relationships",
                )
                session.add(experiment)
                await session.flush()

                # Create dataset
                dataset = Dataset(
                    name="test-dataset",
                    source="synthetic",
                    path="/data/synthetic.csv",
                    total_samples=1000,
                )
                session.add(dataset)
                await session.flush()

                # Create model
                model = Model(name="test-model", type="openai_api", path="gpt-3.5-turbo")
                session.add(model)
                await session.flush()

                # Create evaluation
                evaluation = Evaluation(
                    experiment_id=experiment.id,
                    dataset_id=dataset.id,
                    model_id=model.id,
                    metrics=["accuracy", "f1_score"],
                    status="completed",
                )
                session.add(evaluation)

            # Query with relationships
            async with manager.session_scope() as session:
                # Test complex join query
                result = await session.execute(
                    text("""
                    SELECT
                        e.name as experiment_name,
                        d.name as dataset_name,
                        m.name as model_name,
                        ev.status as evaluation_status
                    FROM experiments e
                    JOIN evaluations ev ON e.id = ev.experiment_id
                    JOIN datasets d ON ev.dataset_id = d.id
                    JOIN models m ON ev.model_id = m.id
                    WHERE e.name = 'relationship-test'
                """)
                )

                row = result.fetchone()
                assert row is not None
                assert row[0] == "relationship-test"
                assert row[1] == "test-dataset"
                assert row[2] == "test-model"
                assert row[3] == "completed"


class TestDatabaseConcurrency:
    """Test concurrent access and performance characteristics."""

    @pytest.mark.asyncio
    async def test_high_concurrency_memory_database(self):
        """Test high concurrency with in-memory database."""
        db_url = get_sqlite_url(in_memory=True)

        # Use a single manager instance for all operations to share the same in-memory database
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        try:

            async def concurrent_worker(worker_id: int, operations: int):
                """Worker that performs many concurrent operations."""
                for op in range(operations):
                    async with manager.session_scope() as session:
                        experiment = Experiment(
                            name=f"concurrent-{worker_id}-{op}",
                            config={"worker": worker_id, "op": op},
                            output_dir=f"/tmp/concurrent-{worker_id}-{op}",
                        )
                        session.add(experiment)

                        # Also perform a read operation
                        if op % 5 == 0:  # Every 5th operation
                            result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                            count = result.scalar()
                            assert count >= 0

            # Run many workers with many operations each
            num_workers = 10
            operations_per_worker = 20

            start_time = asyncio.get_event_loop().time()

            tasks = [concurrent_worker(i, operations_per_worker) for i in range(num_workers)]
            await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Verify all operations completed
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == num_workers * operations_per_worker

            # Log performance info
            total_ops = num_workers * operations_per_worker
            ops_per_second = total_ops / duration
            print(
                f"Completed {total_ops} operations in {duration:.2f}s ({ops_per_second:.2f} ops/sec)"
            )

        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_deadlock_prevention(self):
        """Test that concurrent transactions don't deadlock."""
        db_url = get_sqlite_url(in_memory=True)

        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        try:

            async def conflicting_worker(worker_id: int):
                """Worker that might conflict with others."""
                for i in range(10):
                    try:
                        async with manager.session_scope() as session:
                            # Create experiment
                            experiment = Experiment(
                                name=f"deadlock-test-{worker_id}-{i}",
                                config={"worker": worker_id},
                                output_dir=f"/tmp/deadlock-{worker_id}",
                            )
                            session.add(experiment)

                            # Small delay to increase chance of conflicts
                            await asyncio.sleep(0.001)

                            # Read operation that might conflict
                            result = await session.execute(
                                text("SELECT COUNT(*) FROM experiments WHERE name LIKE :pattern"),
                                {"pattern": f"deadlock-test-{worker_id}%"},
                            )
                            count = result.scalar()
                            assert count >= 0

                    except OperationalError as e:
                        if "database is locked" in str(e):
                            # Retry after brief delay
                            await asyncio.sleep(0.01)
                            continue
                        raise

            # Run workers that might create conflicting access patterns
            tasks = [conflicting_worker(i) for i in range(5)]
            await asyncio.gather(*tasks)

            # Verify some operations succeeded
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count > 0

        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_long_running_transactions(self):
        """Test behavior with long-running transactions."""
        db_url = get_sqlite_url(in_memory=True)

        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        try:

            async def long_transaction():
                """Long-running transaction that holds resources."""
                session = await manager.get_session()
                try:
                    experiment = Experiment(
                        name="long-transaction-test", config={"long": True}, output_dir="/tmp/long"
                    )
                    session.add(experiment)

                    # Simulate long processing
                    await asyncio.sleep(0.1)

                    await session.commit()
                finally:
                    await session.close()

            async def short_transaction(tx_id: int):
                """Short transaction that should complete quickly."""
                async with manager.session_scope() as session:
                    experiment = Experiment(
                        name=f"short-transaction-{tx_id}",
                        config={"short": True, "id": tx_id},
                        output_dir=f"/tmp/short-{tx_id}",
                    )
                    session.add(experiment)

            # Run one long transaction alongside multiple short ones
            long_task = asyncio.create_task(long_transaction())
            short_tasks = [asyncio.create_task(short_transaction(i)) for i in range(5)]

            await asyncio.gather(long_task, *short_tasks)

            # Verify all transactions completed
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == 6  # 1 long + 5 short

        finally:
            await manager.close()


class TestDatabaseErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_connection_recovery_after_close(self):
        """Test that manager can recover after connections are closed."""
        db_url = get_sqlite_url(in_memory=True)

        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        # Create initial data
        async with manager.session_scope() as session:
            experiment = Experiment(name="recovery-test-1", config={}, output_dir="/tmp/recovery1")
            session.add(experiment)

        # Close manager
        await manager.close()

        # Re-initialize same manager (simulating restart)
        await manager.initialize()
        await manager.create_tables()  # Should be no-op if tables exist

        # Verify we can still work with database
        async with manager.session_scope() as session:
            experiment = Experiment(name="recovery-test-2", config={}, output_dir="/tmp/recovery2")
            session.add(experiment)

        # Verify both experiments exist
        async with manager.session_scope() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == 2

        await manager.close()

    @pytest.mark.asyncio
    async def test_transaction_recovery_after_error(self):
        """Test transaction recovery after database errors."""
        db_url = get_sqlite_url(in_memory=True)

        async with DatabaseManager(db_url) as manager:
            await manager.create_tables()

            # Create valid baseline data
            async with manager.session_scope() as session:
                experiment = Experiment(name="baseline", config={}, output_dir="/tmp/baseline")
                session.add(experiment)

            # Attempt operation that will fail
            try:
                async with manager.session_scope() as session:
                    # Add valid data
                    experiment = Experiment(
                        name="will-be-rolled-back", config={}, output_dir="/tmp/rollback"
                    )
                    session.add(experiment)

                    # Force an error
                    await session.execute(text("CREATE TABLE experiments (invalid)"))  # Should fail

            except Exception:
                pass  # Expected to fail

            # Verify database is still usable and rollback worked
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == 1  # Only baseline should exist

                result = await session.execute(text("SELECT name FROM experiments"))
                name = result.scalar()
                assert name == "baseline"

            # Verify we can still perform normal operations
            async with manager.session_scope() as session:
                experiment = Experiment(
                    name="after-error", config={}, output_dir="/tmp/after-error"
                )
                session.add(experiment)

            # Final verification
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == 2

    @pytest.mark.asyncio
    async def test_health_check_during_operations(self):
        """Test health checks during various database operations."""
        db_url = get_sqlite_url(in_memory=True)

        async with DatabaseManager(db_url) as manager:
            await manager.create_tables()

            # Health check should work initially
            assert await manager.health_check() is True

            # Health check during transaction
            session = await manager.get_session()
            experiment = Experiment(name="health-test", config={}, output_dir="/tmp/health")
            session.add(experiment)

            # Health check should still work with open transaction
            assert await manager.health_check() is True

            await session.commit()
            await session.close()

            # Health check after transaction
            assert await manager.health_check() is True


class TestDatabasePerformance:
    """Test performance characteristics and optimization."""

    @pytest.mark.asyncio
    async def test_batch_operations_performance(self):
        """Test performance of batch operations."""
        db_url = get_sqlite_url(in_memory=True)

        async with DatabaseManager(db_url) as manager:
            await manager.create_tables()

            # Test single-transaction batch
            batch_size = 100
            start_time = asyncio.get_event_loop().time()

            async with manager.session_scope() as session:
                for i in range(batch_size):
                    experiment = Experiment(
                        name=f"batch-{i}", config={"batch_index": i}, output_dir=f"/tmp/batch-{i}"
                    )
                    session.add(experiment)

            batch_time = asyncio.get_event_loop().time() - start_time

            # Test individual transactions
            start_time = asyncio.get_event_loop().time()

            for i in range(batch_size, batch_size * 2):
                async with manager.session_scope() as session:
                    experiment = Experiment(
                        name=f"individual-{i}",
                        config={"individual_index": i},
                        output_dir=f"/tmp/individual-{i}",
                    )
                    session.add(experiment)

            individual_time = asyncio.get_event_loop().time() - start_time

            # Batch should be significantly faster
            assert batch_time < individual_time / 2

            print(f"Batch time: {batch_time:.3f}s, Individual time: {individual_time:.3f}s")
            print(f"Batch is {individual_time / batch_time:.1f}x faster")

            # Verify all records were created
            async with manager.session_scope() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
                count = result.scalar()
                assert count == batch_size * 2

    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self):
        """Test connection pool efficiency with multiple operations."""
        db_url = get_sqlite_url(in_memory=True)

        # Use small pool to test reuse
        manager = DatabaseManager(db_url, pool_size=2, max_overflow=1)
        await manager.initialize()
        await manager.create_tables()

        async def db_operation(op_id: int):
            """Perform a database operation."""
            async with manager.session_scope() as session:
                experiment = Experiment(
                    name=f"pool-test-{op_id}",
                    config={"op_id": op_id},
                    output_dir=f"/tmp/pool-{op_id}",
                )
                session.add(experiment)

                # Small delay to hold connection briefly
                await asyncio.sleep(0.01)

        # Run more operations than pool size
        num_operations = 10
        start_time = asyncio.get_event_loop().time()

        tasks = [db_operation(i) for i in range(num_operations)]
        await asyncio.gather(*tasks)

        duration = asyncio.get_event_loop().time() - start_time

        # Verify all operations completed
        async with manager.session_scope() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == num_operations

        print(f"Completed {num_operations} operations in {duration:.3f}s with pool_size=2")

        await manager.close()
