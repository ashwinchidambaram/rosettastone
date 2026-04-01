from logging.config import fileConfig

from sqlmodel import SQLModel

# Import all models so SQLModel.metadata is fully populated before autogenerate.
import rosettastone.server.models  # noqa: F401
from alembic import context
from rosettastone.server.database import get_engine

# Alembic Config object — provides access to values in the .ini file.
config = context.config

# Set up Python logging from the ini file, if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Use SQLModel's shared metadata so alembic can detect all table definitions.
target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with a URL rather than a live Engine.
    Calls to context.execute() emit the given string to the script output.
    """
    url = str(get_engine().url)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Uses the application's own engine (respects DATABASE_URL / ROSETTASTONE_DB_PATH)
    rather than the config-file URL, so the same env vars control both the app and
    alembic.
    """
    connectable = get_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Render the schema for SQLite without the schema prefix
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
