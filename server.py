"""mini-vllm HTTP server — OpenAI-compatible /v1/chat/completions.

Start with:
    python scripts/start_server.py --config configs/default.yaml
"""

import time
import uuid

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from utils.config import GlobalConfig, print_runtime_config
from utils.logger import logger
from engine.loader import load_model
from engine.runtime_setup import apply_runtime_patches
from engine.processor import load_processor
from engine.model_runner import ModelRunner
from engine.async_engine import AsyncLLMEngine
from engine.schema import SamplingParams


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "mini-vllm"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    enable_thinking: Optional[bool] = None  # overrides server config when set


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChoiceDelta(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChoiceDelta]


class ChoiceMessage(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChoiceMessage]
    usage: Usage


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_engine: Optional[AsyncLLMEngine] = None
_cfg: Optional[GlobalConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs at startup; parse args here so uvicorn.run() can still be called
    # from __main__ after argparse.
    global _engine, _cfg
    model, tokenizer = load_model(_cfg)
    model = apply_runtime_patches(model, _cfg)
    processor = load_processor(_cfg)
    runner = ModelRunner(model=model, tokenizer=tokenizer, processor=processor, cfg=_cfg)
    _engine = AsyncLLMEngine(runner)
    logger.info("Server ready.")
    yield
    _engine.shutdown()


app = FastAPI(title="mini-vllm", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    # Build prompt from messages (apply chat template via engine's runner tokenizer)
    runner = _engine._runner
    tokenizer = runner.tokenizer
    has_tmpl = getattr(tokenizer, "chat_template", None) is not None

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    use_thinking = (
        request.enable_thinking
        if request.enable_thinking is not None
        else _cfg.generation.use_thinking
    )
    if has_tmpl:
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=use_thinking,
            tokenize=False,
        )
    else:
        # Fallback: concatenate messages
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens or _cfg.generation.max_new_tokens,
        enable_thinking=use_thinking,
    )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_response(request_id, created, request.model, prompt, sampling_params),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all tokens
    full_text = ""
    finish_reason = "stop"
    async for output in _engine.generate(prompt, sampling_params, request_id):
        full_text += output.text_delta
        if output.finish_reason:
            finish_reason = output.finish_reason

    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=request.model,
        choices=[
            ChoiceMessage(
                index=0,
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(),
    )


async def _stream_response(request_id, created, model_name, prompt, sampling_params):
    # Opening chunk with role
    chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model_name,
        choices=[ChoiceDelta(index=0, delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"

    async for output in _engine.generate(prompt, sampling_params, request_id):
        if output.error:
            break
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                ChoiceDelta(
                    index=0,
                    delta=DeltaMessage(content=output.text_delta),
                    finish_reason=output.finish_reason,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"
