"""The `budget` command (OpenRouter spend/limit for the configured key)."""

from __future__ import annotations

import typer

from twominds import cost as cost_mod

from ._app import app


@app.command()
def budget():
    """Show OpenRouter spend / limit / remaining for $OPENROUTER_API_KEY."""
    from dotenv import load_dotenv

    load_dotenv()
    bal = cost_mod.openrouter_balance()
    if not bal:
        typer.echo(
            "OpenRouter balance unavailable (no OPENROUTER_API_KEY / API error)."
        )
        raise typer.Exit(1)
    lim = bal.get("limit")
    typer.echo("OpenRouter key budget:")
    used = f"${bal.get('usage', 0):.2f}" + (f" / ${lim:.0f} limit" if lim else "")
    typer.echo(f"  usage (this key):     {used}")
    if bal.get("limit_remaining") is not None:
        typer.echo(f"  remaining:            ${bal['limit_remaining']:.2f}")
    typer.echo(
        f"  today/week/month:     ${bal.get('usage_daily', 0):.2f}"
        f" / ${bal.get('usage_weekly', 0):.2f} / ${bal.get('usage_monthly', 0):.2f}"
    )
    cr = cost_mod.openrouter_usage()
    if cr is not None:
        typer.echo(f"  account total_usage:  ${cr:.2f}")
