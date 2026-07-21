"""`twominds db` — the results database (DATABASE_PLAN.md): init, import, stats."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from sqlalchemy import func, select

from twominds import db as db_mod
from twominds.db import importer as importer_mod
from twominds.db import schema

db_app = typer.Typer(
    no_args_is_help=True,
    help="The results database: initialize, backfill from results trees, inspect.",
)

_DB_OPT = typer.Option(
    db_mod.DEFAULT_DB_PATH, "--db", help="Database file (created if missing)."
)


@db_app.command()
def init(db: Path = _DB_OPT) -> None:
    """Create the database (or migrate an existing one to the current schema)."""
    typer.echo(f"database ready: {db_mod.init_db(db)}")


@db_app.command("import")
def import_(
    paths: Optional[list[Path]] = typer.Argument(
        None, help="Results trees to backfill (default: results/)."
    ),
    db: Path = _DB_OPT,
) -> None:
    """Backfill store generations and run dirs into the database (idempotent)."""
    roots = [p for p in (paths or [Path("results")]) if p.exists()]
    if not roots:
        raise typer.BadParameter("no existing paths to import")
    db_mod.init_db(db)
    with db_mod.session(db) as sess:
        stats = importer_mod.import_paths(sess, roots)
    typer.echo(stats.summary())
    for w in stats.warnings[:20]:
        typer.echo(f"warning: {w}")
    if len(stats.warnings) > 20:
        typer.echo(f"... and {len(stats.warnings) - 20} more warnings")


@db_app.command()
def stats(db: Path = _DB_OPT) -> None:
    """Row counts per table, plus the runs the database knows."""
    if not Path(db).exists():
        raise FileNotFoundError(f"no database at {db} (run `twominds db init`)")
    tables = (
        schema.Model,
        schema.Question,
        schema.QuestionVersion,
        schema.Family,
        schema.FamilyVersion,
        schema.JudgeConfig,
        schema.GenBatch,
        schema.Response,
        schema.Bundle,
        schema.Judgment,
        schema.Clustering,
        schema.BundleMetrics,
        schema.FamilyAnalysis,
        schema.Embedding,
        schema.Run,
    )
    with db_mod.session(db) as sess:
        for t in tables:
            count = sess.scalar(select(func.count()).select_from(t))
            typer.echo(f"{t.__tablename__:>18}: {count}")
        runs = sess.scalars(select(schema.Run).order_by(schema.Run.created_at)).all()
        if runs:
            typer.echo("\nruns:")
        for r in runs:
            n_bundles = sess.scalar(
                select(func.count())
                .select_from(schema.RunBundle)
                .where(schema.RunBundle.run_id == r.id)
            )
            typer.echo(
                f"  {r.name}  [{r.kind}]  {r.created_at or '?'}  {n_bundles} bundles"
            )
