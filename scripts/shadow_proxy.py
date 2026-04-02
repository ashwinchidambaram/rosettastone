#!/usr/bin/env python3
"""Shadow deployment proxy — routes traffic to primary and shadow models simultaneously.

Usage:
    uv run python scripts/shadow_proxy.py --config shadow_config.yaml --port 8080
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import typer
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app_proxy = FastAPI(title="RosettaStone Shadow Proxy")
_shadow_config: Any = None  # ShadowConfig instance, set at startup
_log_dir: Path = Path("./shadow_logs/")


@app_proxy.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """Handle chat completions: return primary response, shadow in background."""
    import litellm

    from rosettastone.shadow.log_format import ShadowLogEntry, write_log_entry

    body = await request.json()
    messages = body.get("messages", [])
    prompt_text = " ".join(
        m.get("content", "") for m in messages if isinstance(m.get("content"), str)
    )

    if _shadow_config is None:
        return JSONResponse({"error": "Shadow proxy not configured"}, status_code=503)

    primary_model = (
        _shadow_config.source_model
        if _shadow_config.primary == "source"
        else _shadow_config.target_model
    )
    shadow_model = (
        _shadow_config.target_model
        if _shadow_config.primary == "source"
        else _shadow_config.source_model
    )

    if _shadow_config.optimized_prompt:
        messages = [{"role": "system", "content": _shadow_config.optimized_prompt}] + messages

    primary_response_text = ""
    primary_latency = 0.0
    primary_tokens = 0
    primary_cost = 0.0

    t0 = time.monotonic()
    try:
        primary_resp = await asyncio.to_thread(
            litellm.completion,
            model=primary_model,
            messages=messages,
        )
        primary_latency = (time.monotonic() - t0) * 1000
        primary_response_text = primary_resp.choices[0].message.content or ""
        primary_tokens = getattr(primary_resp.usage, "total_tokens", 0) or 0
        primary_cost = (
            getattr(primary_resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    async def _shadow_call() -> None:
        shadow_text = ""
        shadow_latency = 0.0
        shadow_tokens = 0
        shadow_cost = 0.0
        try:
            ts = time.monotonic()
            shadow_resp = await asyncio.wait_for(
                asyncio.to_thread(litellm.completion, model=shadow_model, messages=messages),
                timeout=30.0,
            )
            shadow_latency = (time.monotonic() - ts) * 1000
            shadow_text = shadow_resp.choices[0].message.content or ""
            shadow_tokens = getattr(shadow_resp.usage, "total_tokens", 0) or 0
            shadow_cost = (
                getattr(shadow_resp, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
            )
        except Exception:
            pass  # Shadow failures must never surface to caller

        source_resp = primary_response_text if _shadow_config.primary == "source" else shadow_text
        target_resp = shadow_text if _shadow_config.primary == "source" else primary_response_text

        entry = ShadowLogEntry(
            request_id=str(uuid.uuid4()),
            prompt=prompt_text,
            source_model=_shadow_config.source_model,
            target_model=_shadow_config.target_model,
            source_response=source_resp,
            target_response=target_resp,
            source_latency_ms=primary_latency
            if _shadow_config.primary == "source"
            else shadow_latency,
            target_latency_ms=shadow_latency
            if _shadow_config.primary == "source"
            else primary_latency,
            source_tokens=primary_tokens if _shadow_config.primary == "source" else shadow_tokens,
            target_tokens=shadow_tokens if _shadow_config.primary == "source" else primary_tokens,
            source_cost=primary_cost if _shadow_config.primary == "source" else shadow_cost,
            target_cost=shadow_cost if _shadow_config.primary == "source" else primary_cost,
        )
        write_log_entry(entry, _log_dir)

    asyncio.create_task(_shadow_call())

    return JSONResponse(
        {
            "choices": [{"message": {"role": "assistant", "content": primary_response_text}}],
            "model": primary_model,
            "usage": {"total_tokens": primary_tokens},
        }
    )


cli = typer.Typer()


@cli.command()
def main(
    config: Path = typer.Option(Path("shadow_config.yaml"), "--config", help="Shadow config YAML"),
    port: int = typer.Option(8080, "--port", help="Port to listen on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),  # noqa: S104
) -> None:
    """Start the RosettaStone shadow proxy."""
    import uvicorn

    from rosettastone.shadow.config import ShadowConfig

    global _shadow_config, _log_dir

    config_data = yaml.safe_load(config.read_text())
    _shadow_config = ShadowConfig.model_validate(config_data)
    _log_dir = Path(_shadow_config.log_path)

    typer.echo(f"Shadow proxy starting on {host}:{port}")
    typer.echo(
        f"Primary: {_shadow_config.source_model if _shadow_config.primary == 'source' else _shadow_config.target_model}"
    )
    typer.echo(
        f"Shadow: {_shadow_config.target_model if _shadow_config.primary == 'source' else _shadow_config.source_model}"
    )
    typer.echo(f"Logging to: {_log_dir}")

    uvicorn.run(app_proxy, host=host, port=port)


if __name__ == "__main__":
    cli()
