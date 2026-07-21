"""Engine, session, and migration entry points for the results database."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

DEFAULT_DB_PATH = Path("results/twominds.db")
_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _url(path: Path | str) -> str:
    return f"sqlite:///{Path(path).resolve()}"


def make_engine(path: Path | str = DEFAULT_DB_PATH) -> Engine:
    engine = create_engine(_url(path))

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    return engine


def _alembic_config(path: Path | str):
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", _url(path))
    return cfg


def init_db(path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Create the database if missing, then migrate it to the current schema.
    Safe to call before every use."""
    from alembic import command

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(_alembic_config(path), "head")
    return path


@contextmanager
def session(path: Path | str = DEFAULT_DB_PATH):
    engine = make_engine(path)
    try:
        with Session(engine) as sess:
            yield sess
    finally:
        engine.dispose()
