"""
Database manager for the LLM Cybersecurity Benchmark system.

This module provides async database connection management, session handling,
and transaction support with both SQLite and PostgreSQL compatibility.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import event, text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool, StaticPool

from .database import Base

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""

    pass


class DatabaseInitializationError(Exception):
    """Raised when database initialization fails."""

    pass


class DatabaseManager:
    """
    Async database manager with connection pooling and session management.

    Provides comprehensive database management including connections, sessions,
    transactions, migrations, and health monitoring for both development
    (SQLite) and production (PostgreSQL) environments.
    """

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        connect_timeout: int = 10,
    ) -> None:
        """
        Initialize database manager.

        Args:
            database_url: Database connection URL
            echo: Whether to echo SQL statements (for debugging)
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum overflow connections beyond pool_size
            pool_timeout: Timeout in seconds to get connection from pool
            pool_recycle: Recycle connections after this many seconds
            connect_timeout: Connection timeout in seconds
        """
        self.database_url = database_url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.connect_timeout = connect_timeout

        # Parse database URL to determine database type
        parsed = urlparse(database_url)
        self.db_type = parsed.scheme.split("+")[0] if "+" in parsed.scheme else parsed.scheme

        # Initialize engine and session factory
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._initialized = False
        self._closed = False

        logger.info(f"DatabaseManager initialized for {self.db_type} database")

    @property
    def engine(self) -> AsyncEngine:
        """Get the async database engine."""
        if self._engine is None:
            raise DatabaseConnectionError(
                "Database engine not initialized. Call initialize() first."
            )
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the async session factory."""
        if self._session_factory is None:
            raise DatabaseConnectionError(
                "Session factory not initialized. Call initialize() first."
            )
        return self._session_factory

    def _configure_engine_options(self) -> dict[str, Any]:
        """Configure engine options based on database type."""
        base_options = {
            "echo": self.echo,
            "future": True,  # Use SQLAlchemy 2.0 style
            "connect_args": {"timeout": self.connect_timeout},
        }

        if self.db_type == "sqlite":
            # SQLite-specific configuration
            base_options.update(
                {
                    "poolclass": StaticPool if ":memory:" in self.database_url else NullPool,
                    "connect_args": {
                        "timeout": self.connect_timeout,
                        "check_same_thread": False,  # Allow sharing connection between threads
                    },
                }
            )
        elif self.db_type == "postgresql":
            # PostgreSQL-specific configuration
            base_options.update(
                {
                    "poolclass": QueuePool,
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "pool_recycle": self.pool_recycle,
                    "pool_pre_ping": True,  # Validate connections before use
                    "connect_args": {
                        "connect_timeout": self.connect_timeout,
                        "server_settings": {
                            "application_name": "llm_cybersec_benchmark",
                        },
                    },
                }
            )
        else:
            # Generic configuration for other databases
            base_options.update(
                {
                    "poolclass": QueuePool,
                    "pool_size": self.pool_size,
                    "max_overflow": self.max_overflow,
                    "pool_timeout": self.pool_timeout,
                    "pool_recycle": self.pool_recycle,
                }
            )

        return base_options

    async def initialize(self) -> None:
        """
        Initialize database engine and session factory.

        Raises:
            DatabaseConnectionError: If connection fails
            DatabaseInitializationError: If initialization fails
        """
        if self._initialized:
            logger.warning("DatabaseManager already initialized")
            return

        if self._closed:
            # Reset the closed state to allow re-initialization
            self._closed = False

        try:
            # Create async engine with appropriate configuration
            engine_options = self._configure_engine_options()
            self._engine = create_async_engine(self.database_url, **engine_options)

            # Set up SQLite-specific event listeners
            if self.db_type == "sqlite":
                self._setup_sqlite_events()

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,  # Keep objects usable after commit
                autoflush=True,
                autocommit=False,
            )

            # Test connection
            await self._test_connection()

            self._initialized = True
            logger.info("DatabaseManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            await self._cleanup_on_error()
            raise DatabaseInitializationError(f"Database initialization failed: {e}") from e

    def _setup_sqlite_events(self) -> None:
        """Set up SQLite-specific event listeners."""
        if self._engine is None:
            return

        # Enable foreign key constraints for SQLite
        @event.listens_for(self._engine.sync_engine, "connect")  # type: ignore
        def set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            # Set WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            # Optimize performance
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA temp_store=memory")
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
            cursor.close()

    async def _test_connection(self) -> None:
        """Test database connection."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.debug("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise DatabaseConnectionError(f"Cannot connect to database: {e}") from e

    async def _cleanup_on_error(self) -> None:
        """Clean up resources on initialization error."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
        self._session_factory = None

    async def create_tables(self) -> None:
        """
        Create all database tables.

        Raises:
            DatabaseInitializationError: If table creation fails
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise DatabaseInitializationError(f"Table creation failed: {e}") from e

    async def drop_tables(self) -> None:
        """
        Drop all database tables.

        WARNING: This will delete all data!

        Raises:
            DatabaseInitializationError: If table dropping fails
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.warning("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise DatabaseInitializationError(f"Table dropping failed: {e}") from e

    async def get_session(self) -> AsyncSession:
        """
        Get a new database session.

        Returns:
            AsyncSession: Database session

        Raises:
            DatabaseConnectionError: If manager not initialized
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        return self.session_factory()

    @asynccontextmanager
    async def session_scope(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide a transactional scope around a series of operations.

        This context manager will automatically commit the transaction if
        no exceptions occur, or rollback if an exception is raised.

        Yields:
            AsyncSession: Database session with automatic transaction management

        Example:
            async with db_manager.session_scope() as session:
                experiment = Experiment(name="test")
                session.add(experiment)
                # Automatically committed on success, rolled back on error
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            await session.close()

    async def health_check(self) -> bool:
        """
        Perform a health check on the database connection.

        Returns:
            bool: True if database is healthy, False otherwise
        """
        if not self._initialized or self._closed:
            return False

        try:
            async with self.session_scope() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def get_connection_info(self) -> dict[str, Any]:
        """
        Get information about database connections and pool status.

        Returns:
            dict: Connection information and pool statistics
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        info = {
            "status": "initialized",
            "database_type": self.db_type,
            "database_url": str(self.database_url).split("@")[-1]
            if "@" in str(self.database_url)
            else str(self.database_url),  # Hide credentials
            "echo": self.echo,
        }

        # Add pool information for pooled connections
        if hasattr(self.engine.pool, "size"):
            pool = self.engine.pool
            info.update(
                {
                    "pool_size": getattr(pool, "size", lambda: "N/A")(),
                    "checked_in": getattr(pool, "checkedin", lambda: "N/A")(),
                    "checked_out": getattr(pool, "checkedout", lambda: "N/A")(),
                    "overflow": getattr(pool, "overflow", lambda: "N/A")(),
                    "invalid": getattr(pool, "invalid", lambda: "N/A")(),
                }
            )

        return info

    async def execute_raw_sql(self, sql: str, parameters: dict[str, Any] | None = None) -> Any:
        """
        Execute raw SQL query.

        Args:
            sql: SQL query string
            parameters: Optional query parameters

        Returns:
            Query result

        Raises:
            DatabaseError: If query execution fails
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        try:
            async with self.session_scope() as session:
                result = await session.execute(text(sql), parameters or {})
                return result
        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise DatabaseError(f"SQL execution failed: {e}", {}, e) from e

    async def vacuum_database(self) -> None:
        """
        Perform database maintenance (VACUUM for SQLite, ANALYZE for others).

        Raises:
            DatabaseError: If maintenance operation fails
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        try:
            if self.db_type == "sqlite":
                # SQLite VACUUM
                async with self.engine.begin() as conn:
                    await conn.execute(text("VACUUM"))
                logger.info("SQLite database vacuumed successfully")
            else:
                # PostgreSQL ANALYZE
                async with self.engine.begin() as conn:
                    await conn.execute(text("ANALYZE"))
                logger.info("Database analyzed successfully")
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            raise DatabaseError(f"Database maintenance failed: {e}", {}, e) from e

    async def backup_database(self, backup_path: str) -> None:
        """
        Create a backup of the database (SQLite only).

        Args:
            backup_path: Path where backup should be saved

        Raises:
            DatabaseError: If backup fails
            NotImplementedError: For non-SQLite databases
        """
        if not self._initialized:
            raise DatabaseConnectionError("Database manager not initialized")

        if self.db_type != "sqlite":
            raise NotImplementedError("Backup is currently only supported for SQLite databases")

        try:
            import shutil

            # Extract database path from URL
            db_url_str = str(self.database_url)
            db_path = db_url_str.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")

            if ":memory:" in db_path:
                raise DatabaseError(
                    "Cannot backup in-memory database",
                    {},
                    RuntimeError("In-memory database cannot be backed up"),
                )

            # Copy database file
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")

        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise DatabaseError(f"Backup failed: {e}", {}, e) from e

    async def close(self) -> None:
        """
        Close database connections and clean up resources.
        """
        if self._closed:
            logger.warning("DatabaseManager already closed")
            return

        try:
            if self._engine is not None:
                await self._engine.dispose()
                logger.info("Database engine disposed")
        except Exception as e:
            logger.error(f"Error disposing database engine: {e}")
        finally:
            self._engine = None
            self._session_factory = None
            self._initialized = False
            self._closed = True
            logger.info("DatabaseManager closed successfully")

    async def __aenter__(self) -> "DatabaseManager":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        if self._closed:
            status = "closed"
        return f"<DatabaseManager(db_type='{self.db_type}', status='{status}')>"


# Utility functions for common database operations
async def create_database_manager(
    database_url: str, echo: bool = False, **kwargs: Any
) -> DatabaseManager:
    """
    Create and initialize a database manager.

    Args:
        database_url: Database connection URL
        echo: Whether to echo SQL statements
        **kwargs: Additional arguments for DatabaseManager

    Returns:
        DatabaseManager: Initialized database manager
    """
    manager = DatabaseManager(database_url, echo=echo, **kwargs)
    await manager.initialize()
    return manager


async def ensure_database_ready(manager: DatabaseManager) -> None:
    """
    Ensure database is ready by creating tables if needed.

    Args:
        manager: Database manager instance
    """
    if not await manager.health_check():
        raise DatabaseConnectionError("Database is not healthy")

    # Create tables if they don't exist
    await manager.create_tables()


# Database URL helpers
def get_sqlite_url(db_path: str = "benchmark.db", in_memory: bool = False) -> str:
    """
    Get SQLite database URL.

    Args:
        db_path: Path to database file
        in_memory: Use in-memory database

    Returns:
        str: SQLite database URL
    """
    if in_memory:
        return "sqlite+aiosqlite:///:memory:"
    return f"sqlite+aiosqlite:///{db_path}"


def get_postgresql_url(
    username: str,
    password: str,
    host: str,
    port: int = 5432,
    database: str = "benchmark",
) -> str:
    """
    Get PostgreSQL database URL.

    Args:
        username: Database username
        password: Database password
        host: Database host
        port: Database port
        database: Database name

    Returns:
        str: PostgreSQL database URL
    """
    return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
