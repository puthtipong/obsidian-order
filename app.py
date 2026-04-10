"""
Obsidian Order Interro-Helper — FastAPI backend.

Start:
    export OPENAI_API_KEY=sk-...
    uvicorn app:app --reload --port 8765

Then open http://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional

from transforms import TRANSFORMS, apply_custom, apply_translation

from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    # Boot-time config from env vars (optional — can be overridden via /config)
    api_key = os.environ.get("OPENAI_API_KEY")
    endpoint = os.getenv("OPENAI_CHAT_ENDPOINT", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4")
    reasoning_model = os.getenv("OPENAI_REASONING_MODEL", "gpt-5.4")

    app.state.config = {
        "endpoint": endpoint,
        "model": model,
        "reasoning_model": reasoning_model,
        "reasoning_effort": "medium",
    }

    if api_key:
        _init_llm_targets(app, api_key, endpoint, model, reasoning_model, "medium")
        app.state.llm_available = True
    else:
        app.state.llm = None
        app.state.llm_reasoning = None
        app.state.llm_available = False

    yield


def _init_llm_targets(
    app: FastAPI,
    api_key: str,
    endpoint: str,
    model: str,
    reasoning_model: str,
    reasoning_effort: Optional[str],
) -> None:
    app.state.llm = OpenAIChatTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=model,
    )
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort and reasoning_effort != "none" else None
    app.state.llm_reasoning = OpenAIChatTarget(
        endpoint=endpoint,
        api_key=api_key,
        model_name=reasoning_model,
        extra_body_parameters=extra,
    )
    app.state.llm_available = True


app = FastAPI(title="Prompt Mutation Helper", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ConfigRequest(BaseModel):
    api_key: str
    endpoint: str = "https://api.openai.com/v1"
    model: str = "gpt-5.4"
    reasoning_model: str = "gpt-5.4"
    reasoning_effort: Optional[str] = "medium"  # "none" | "low" | "medium" | "high"


class TransformRequest(BaseModel):
    prompt: str
    selected: list[str] = []
    languages: list[str] = []
    context: str = ""
    custom_prompt: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/transforms")
async def list_transforms():
    """Return all registered transforms as a JSON list (used by UI to build checkboxes)."""
    return [t.to_dict() for t in TRANSFORMS.values()]


@app.get("/status")
async def status():
    cfg = getattr(app.state, "config", {})
    return {
        "llm_available": app.state.llm_available,
        "model": cfg.get("model"),
        "reasoning_model": cfg.get("reasoning_model"),
        "reasoning_effort": cfg.get("reasoning_effort"),
    }


@app.post("/config")
async def update_config(req: ConfigRequest):
    """Apply new API key / model config at runtime without restarting."""
    _init_llm_targets(
        app,
        req.api_key,
        req.endpoint,
        req.model,
        req.reasoning_model,
        req.reasoning_effort,
    )
    app.state.config = {
        "endpoint": req.endpoint,
        "model": req.model,
        "reasoning_model": req.reasoning_model,
        "reasoning_effort": req.reasoning_effort,
    }
    return {
        "ok": True,
        "model": req.model,
        "reasoning_model": req.reasoning_model,
        "reasoning_effort": req.reasoning_effort,
    }


@app.post("/transform")
async def run_transforms(req: TransformRequest):
    """
    Run all selected transforms + translations in parallel and stream results
    back as Server-Sent Events.

    Each SSE event is a JSON object:
        {"id": "base64",  "label": "Base64",   "group": "encoding",  "result": "...", "error": null}
        {"id": "lang-fr", "label": "→ French", "group": "translation","result": "...", "error": null}
        {"done": true}   ← final sentinel
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    unknown = [s for s in req.selected if s not in TRANSFORMS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown transform IDs: {unknown}")

    llm = app.state.llm
    llm_reasoning = app.state.llm_reasoning

    async def run_one_transform(tid: str, queue: asyncio.Queue) -> None:
        t = TRANSFORMS[tid]
        try:
            result = await t.apply_async(req.prompt, llm, req.context)
            await queue.put({"id": tid, "label": t.label, "group": t.group,
                             "result": result, "error": None})
        except Exception as exc:
            await queue.put({"id": tid, "label": t.label, "group": t.group,
                             "result": None, "error": str(exc)})

    async def run_one_language(lang: str, queue: asyncio.Queue) -> None:
        tid = f"lang-{lang}"
        try:
            result = await apply_translation(lang, req.prompt, llm, req.context)
            await queue.put({"id": tid, "label": f"→ {lang}", "group": "translation",
                             "result": result, "error": None})
        except Exception as exc:
            await queue.put({"id": tid, "label": f"→ {lang}", "group": "translation",
                             "result": None, "error": str(exc)})

    async def run_one_custom(queue: asyncio.Queue) -> None:
        try:
            result = await apply_custom(req.custom_prompt, req.prompt, llm_reasoning, req.context)
            await queue.put({"id": "custom", "label": "Custom Transform", "group": "custom",
                             "result": result, "error": None})
        except Exception as exc:
            await queue.put({"id": "custom", "label": "Custom Transform", "group": "custom",
                             "result": None, "error": str(exc)})

    async def event_stream():
        has_custom = bool(req.custom_prompt.strip())
        total = len(req.selected) + len(req.languages) + (1 if has_custom else 0)
        if total == 0:
            yield 'data: {"done": true}\n\n'
            return

        queue: asyncio.Queue = asyncio.Queue()

        tasks = [
            asyncio.create_task(run_one_transform(tid, queue))
            for tid in req.selected
        ] + [
            asyncio.create_task(run_one_language(lang, queue))
            for lang in req.languages
        ] + (
            [asyncio.create_task(run_one_custom(queue))] if has_custom else []
        )

        for _ in range(total):
            item = await queue.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        yield 'data: {"done": true}\n\n'

        # Ensure all tasks are awaited even if already done
        await asyncio.gather(*tasks, return_exceptions=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
