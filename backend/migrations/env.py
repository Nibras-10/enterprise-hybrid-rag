"""Alembic environment configured from the Supabase database URL."""

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from app.models import Base

config = context.config
load_dotenv()
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.getenv("SUPABASE_DATABASE_URL")
if database_url:
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
elif not config.get_main_option("sqlalchemy.url"):
    raise RuntimeError("Set SUPABASE_DATABASE_URL before running database migrations.")

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Render migrations without opening a database connection."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the configured database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
