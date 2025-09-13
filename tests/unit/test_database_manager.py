"""
Unit tests for database manager.

Tests database connection management, session handling, and transaction support.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from benchmark.core.database import Experiment
from benchmark.core.database_manager import (
    DatabaseConnectionError,
    DatabaseInitializationError,
    DatabaseManager,
    create_database_manager,
    ensure_database_ready,
    get_postgresql_url,
    get_sqlite_url,
)


class TestDatabaseManager:
    """Test DatabaseManager functionality."""

    def test_init_sqlite(self):
        """Test DatabaseManager initialization with SQLite URL."""
        db_url = "sqlite+aiosqlite:///test.db"
        manager = DatabaseManager(db_url)

        assert manager.database_url == db_url
        assert manager.db_type == "sqlite"
        assert not manager._initialized
        assert not manager._closed

    def test_init_postgresql(self):
        """Test DatabaseManager initialization with PostgreSQL URL."""
        db_url = "postgresql+asyncpg://user:pass@localhost:5432/db"
        manager = DatabaseManager(db_url)

        assert manager.database_url == db_url
        assert manager.db_type == "postgresql"
        assert not manager._initialized
        assert not manager._closed

    def test_init_with_custom_parameters(self):
        """Test DatabaseManager initialization with custom parameters."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(
            db_url,
            echo=True,
            pool_size=10,
            max_overflow=20,
            pool_timeout=60,
            pool_recycle=1800,
            connect_timeout=5,
        )

        assert manager.echo is True
        assert manager.pool_size == 10
        assert manager.max_overflow == 20
        assert manager.pool_timeout == 60
        assert manager.pool_recycle == 1800
        assert manager.connect_timeout == 5

    def test_properties_before_initialization(self):
        """Test that properties raise errors before initialization."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(DatabaseConnectionError):
            _ = manager.engine

        with pytest.raises(DatabaseConnectionError):
            _ = manager.session_factory

    def test_configure_engine_options_sqlite(self):
        """Test engine configuration for SQLite."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        options = manager._configure_engine_options()

        assert options["echo"] is False
        assert options["future"] is True
        assert "check_same_thread" in options["connect_args"]
        assert options["connect_args"]["check_same_thread"] is False

    def test_configure_engine_options_postgresql(self):
        """Test engine configuration for PostgreSQL."""
        manager = DatabaseManager(
            "postgresql+asyncpg://user:pass@localhost:5432/db",
            pool_size=5,
            max_overflow=10,
        )
        options = manager._configure_engine_options()

        assert options["pool_size"] == 5
        assert options["max_overflow"] == 10
        assert options["pool_pre_ping"] is True
        assert "application_name" in options["connect_args"]["server_settings"]

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful database manager initialization."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)

        with patch.object(manager, "_test_connection", new_callable=AsyncMock) as mock_test:
            mock_test.return_value = None

            await manager.initialize()

            assert manager._initialized is True
            assert manager._engine is not None
            assert manager._session_factory is not None
            mock_test.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        manager._initialized = True

        with patch("benchmark.core.database_manager.logger") as mock_logger:
            await manager.initialize()
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_after_close(self):
        """Test re-initialization after manager has been closed."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)

        # Initialize first time
        await manager.initialize()
        assert manager._initialized

        # Close the manager
        await manager.close()
        assert manager._closed
        assert not manager._initialized

        # Re-initialize should work (reset _closed state)
        await manager.initialize()
        assert manager._initialized
        assert not manager._closed

        # Clean up
        await manager.close()

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self):
        """Test initialization failure due to connection error."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)

        with patch.object(manager, "_test_connection", new_callable=AsyncMock) as mock_test:
            mock_test.side_effect = OperationalError("Connection failed", None, None)

            with pytest.raises(DatabaseInitializationError):
                await manager.initialize()

            assert manager._initialized is False
            assert manager._engine is None

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        # _test_connection is called during initialization, so if we get here it worked
        assert manager._initialized is True

    @pytest.mark.asyncio
    async def test_create_tables_success(self):
        """Test successful table creation."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        await manager.create_tables()
        # No exception means success

    @pytest.mark.asyncio
    async def test_create_tables_not_initialized(self):
        """Test table creation when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(DatabaseConnectionError):
            await manager.create_tables()

    @pytest.mark.asyncio
    async def test_drop_tables_success(self):
        """Test successful table dropping."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        await manager.drop_tables()
        # No exception means success

    @pytest.mark.asyncio
    async def test_get_session_success(self):
        """Test successful session creation."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        session = await manager.get_session()
        assert isinstance(session, AsyncSession)
        await session.close()

    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self):
        """Test session creation when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(DatabaseConnectionError):
            await manager.get_session()

    @pytest.mark.asyncio
    async def test_session_scope_success(self):
        """Test successful session scope with automatic commit."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        experiment_name = "test-experiment"

        async with manager.session_scope() as session:
            experiment = Experiment(name=experiment_name, config={}, output_dir="/tmp/test")
            session.add(experiment)
            # Should auto-commit on successful exit

        # Verify the experiment was committed
        async with manager.session_scope() as session:
            result = await session.get(Experiment, 1)  # First experiment should have ID 1
            assert result is not None
            assert result.name == experiment_name

    @pytest.mark.asyncio
    async def test_session_scope_rollback_on_error(self):
        """Test session scope with automatic rollback on error."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        with pytest.raises(ValueError):
            async with manager.session_scope() as session:
                experiment = Experiment(name="test-experiment", config={}, output_dir="/tmp/test")
                session.add(experiment)
                # Force an error to test rollback
                raise ValueError("Test error")

        # Verify nothing was committed due to rollback
        async with manager.session_scope() as session:
            result = await session.get(Experiment, 1)
            assert result is None

    @pytest.mark.asyncio
    async def test_session_scope_not_initialized(self):
        """Test session scope when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(DatabaseConnectionError):
            async with manager.session_scope():
                pass

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        result = await manager.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        result = await manager.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_closed(self):
        """Test health check when closed."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.close()

        result = await manager.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_database_error(self):
        """Test health check with database error."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        # Mock session to raise error
        with patch.object(manager, "session_scope") as mock_scope:
            mock_scope.side_effect = OperationalError("DB Error", None, None)

            result = await manager.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_get_connection_info_not_initialized(self):
        """Test connection info when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        info = await manager.get_connection_info()
        assert info["status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_get_connection_info_initialized(self):
        """Test connection info when initialized."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url, echo=True)
        await manager.initialize()

        info = await manager.get_connection_info()
        assert info["status"] == "initialized"
        assert info["database_type"] == "sqlite"
        assert info["echo"] is True
        assert ":memory:" in info["database_url"]

    @pytest.mark.asyncio
    async def test_execute_raw_sql_success(self):
        """Test successful raw SQL execution."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        result = await manager.execute_raw_sql("SELECT 1 as test_value")
        row = result.fetchone()
        assert row[0] == 1

    @pytest.mark.asyncio
    async def test_execute_raw_sql_with_parameters(self):
        """Test raw SQL execution with parameters."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        result = await manager.execute_raw_sql("SELECT :value as test_value", {"value": 42})
        row = result.fetchone()
        assert row[0] == 42

    @pytest.mark.asyncio
    async def test_execute_raw_sql_not_initialized(self):
        """Test raw SQL execution when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(DatabaseConnectionError):
            await manager.execute_raw_sql("SELECT 1")

    @pytest.mark.asyncio
    async def test_vacuum_database_sqlite(self):
        """Test database vacuum for SQLite."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        await manager.vacuum_database()
        # No exception means success

    @pytest.mark.asyncio
    async def test_vacuum_database_postgresql(self):
        """Test database analyze for PostgreSQL."""
        db_url = "postgresql+asyncpg://user:pass@localhost:5432/db"
        manager = DatabaseManager(db_url)

        # Mock engine to avoid actual connection
        mock_engine = Mock()
        mock_conn = AsyncMock()

        # Create an async context manager mock
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_engine.begin.return_value = mock_transaction

        manager._engine = mock_engine
        manager._initialized = True

        await manager.vacuum_database()
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_vacuum_database_not_initialized(self):
        """Test database vacuum when not initialized."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        with pytest.raises(DatabaseConnectionError):
            await manager.vacuum_database()

    @pytest.mark.asyncio
    async def test_backup_database_not_sqlite(self):
        """Test backup with non-SQLite database."""
        db_url = "postgresql+asyncpg://user:pass@localhost:5432/db"
        manager = DatabaseManager(db_url)
        manager._initialized = True

        with pytest.raises(NotImplementedError):
            await manager.backup_database("/tmp/backup.db")

    @pytest.mark.asyncio
    async def test_backup_database_memory(self):
        """Test backup with in-memory database."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        manager._initialized = True

        with pytest.raises(DatabaseError):
            await manager.backup_database("/tmp/backup.db")

    @pytest.mark.asyncio
    async def test_close_success(self):
        """Test successful manager close."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        await manager.close()

        assert manager._closed is True
        assert manager._initialized is False
        assert manager._engine is None
        assert manager._session_factory is None

    @pytest.mark.asyncio
    async def test_close_already_closed(self):
        """Test closing already closed manager."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.close()

        with patch("benchmark.core.database_manager.logger") as mock_logger:
            await manager.close()
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_engine_error(self):
        """Test close with engine disposal error."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        # Replace the engine with a mock that has a dispose method that raises an error
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock(side_effect=Exception("Dispose error"))
        manager._engine = mock_engine

        await manager.close()

        # Should still mark as closed despite error
        assert manager._closed is True

        # Verify dispose was called
        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager functionality."""
        db_url = "sqlite+aiosqlite:///:memory:"

        async with DatabaseManager(db_url) as manager:
            assert manager._initialized is True
            assert await manager.health_check() is True

        # Should be closed after context exit
        assert manager._closed is True

    def test_repr(self):
        """Test string representation."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")

        # Not initialized
        repr_str = repr(manager)
        assert "not_initialized" in repr_str
        assert "sqlite" in repr_str

        # Closed
        manager._closed = True
        repr_str = repr(manager)
        assert "closed" in repr_str


class TestDatabaseManagerUtilities:
    """Test utility functions."""

    @pytest.mark.asyncio
    async def test_create_database_manager(self):
        """Test database manager creation utility."""
        db_url = "sqlite+aiosqlite:///:memory:"

        manager = await create_database_manager(db_url, echo=True)

        assert manager._initialized is True
        assert manager.echo is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_ensure_database_ready_healthy(self):
        """Test ensure database ready with healthy database."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()

        await ensure_database_ready(manager)
        # No exception means success

        await manager.close()

    @pytest.mark.asyncio
    async def test_ensure_database_ready_unhealthy(self):
        """Test ensure database ready with unhealthy database."""
        manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
        # Don't initialize to make it unhealthy

        with pytest.raises(DatabaseConnectionError):
            await ensure_database_ready(manager)

    def test_get_sqlite_url_file(self):
        """Test SQLite URL generation for file database."""
        url = get_sqlite_url("test.db")
        assert url == "sqlite+aiosqlite:///test.db"

    def test_get_sqlite_url_memory(self):
        """Test SQLite URL generation for in-memory database."""
        url = get_sqlite_url(in_memory=True)
        assert url == "sqlite+aiosqlite:///:memory:"

    def test_get_postgresql_url(self):
        """Test PostgreSQL URL generation."""
        url = get_postgresql_url(
            username="user", password="pass", host="localhost", port=5432, database="testdb"
        )
        expected = "postgresql+asyncpg://user:pass@localhost:5432/testdb"
        assert url == expected

    def test_get_postgresql_url_custom_port(self):
        """Test PostgreSQL URL generation with custom port."""
        url = get_postgresql_url(
            username="user", password="pass", host="example.com", port=5433, database="mydb"
        )
        expected = "postgresql+asyncpg://user:pass@example.com:5433/mydb"
        assert url == expected


class TestDatabaseManagerIntegration:
    """Integration tests for database manager with real operations."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete database manager workflow."""
        db_url = "sqlite+aiosqlite:///:memory:"

        async with DatabaseManager(db_url) as manager:
            # Create tables
            await manager.create_tables()

            # Verify health
            assert await manager.health_check() is True

            # Create and query data
            async with manager.session_scope() as session:
                experiment = Experiment(
                    name="integration-test", config={"test": True}, output_dir="/tmp/test"
                )
                session.add(experiment)

            # Query data
            async with manager.session_scope() as session:
                result = await session.get(Experiment, 1)
                assert result is not None
                assert result.name == "integration-test"
                assert result.config == {"test": True}

    @pytest.mark.asyncio
    async def test_multiple_sessions_concurrent(self):
        """Test multiple concurrent sessions."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        async def create_experiment(name: str):
            async with manager.session_scope() as session:
                experiment = Experiment(
                    name=name, config={"concurrent": True}, output_dir=f"/tmp/{name}"
                )
                session.add(experiment)

        # Create multiple experiments concurrently
        tasks = [create_experiment(f"exp-{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify all were created
        async with manager.session_scope() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == 5

        await manager.close()

    @pytest.mark.asyncio
    async def test_transaction_isolation(self):
        """Test transaction isolation between sessions."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        # Session 1: Add experiment but don't commit yet
        session1 = await manager.get_session()
        experiment = Experiment(name="isolation-test", config={}, output_dir="/tmp/test")
        session1.add(experiment)

        # Session 2: Should not see uncommitted data
        async with manager.session_scope() as session2:
            result = await session2.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == 0

        # Commit session 1
        await session1.commit()
        await session1.close()

        # Session 3: Should now see committed data
        async with manager.session_scope() as session3:
            result = await session3.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == 1

        await manager.close()

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and connection resilience."""
        db_url = "sqlite+aiosqlite:///:memory:"
        manager = DatabaseManager(db_url)
        await manager.initialize()
        await manager.create_tables()

        # Create valid data first
        async with manager.session_scope() as session:
            experiment = Experiment(name="valid-experiment", config={}, output_dir="/tmp/valid")
            session.add(experiment)

        # Try to create invalid data (should fail and rollback)
        with pytest.raises(OperationalError):
            async with manager.session_scope() as session:
                # This should succeed
                experiment2 = Experiment(
                    name="invalid-experiment", config={}, output_dir="/tmp/invalid"
                )
                session.add(experiment2)

                # Force an error
                await session.execute(text("INVALID SQL"))

        # Verify the valid data is still there and invalid data was rolled back
        async with manager.session_scope() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM experiments"))
            count = result.scalar()
            assert count == 1  # Only the valid experiment

            result = await session.execute(text("SELECT name FROM experiments"))
            name = result.scalar()
            assert name == "valid-experiment"

        await manager.close()
