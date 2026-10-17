import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = (
    Path(__file__).resolve().parent
    / "qwen2.5-1.5b-anti-sycophancy-lora"
)
MODEL_MODE = os.getenv("MODEL_MODE", "auto").lower()

SYSTEM_PROMPT = """
You are an assistant that values truth, sound reasoning, and calibrated
uncertainty over agreement.

Independently assess the user's claims. Do not change a correct answer
merely because the user disagrees. Acknowledge genuine mistakes, agree
when the user is right, and respectfully explain disagreements when
the user is wrong. Do not be contrarian just for the sake of it.
""".strip()

INFERENCE_LOCK = threading.Lock()


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    message: str | None = Field(default=None, max_length=12000)
    messages: list[Message] | None = Field(default=None, max_length=30)
    max_new_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.2, ge=0, le=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_MODE not in {"auto", "base", "lora"}:
        raise RuntimeError("MODEL_MODE must be auto, base, or lora.")

    adapter_exists = (ADAPTER_DIR / "adapter_config.json").is_file()

    if MODEL_MODE == "lora" and not adapter_exists:
        raise FileNotFoundError(
            f"LoRA adapter not found: {ADAPTER_DIR}"
        )

    use_lora = MODEL_MODE == "lora" or (
        MODEL_MODE == "auto" and adapter_exists
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device == "cuda"
        else torch.float32
    )

    print("Loading base model:", BASE_MODEL)
    print("Device:", device)
    print("Mode:", "LoRA adapter" if use_lora else "base model")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if use_lora:
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    else:
        model = base

    model.to(device)
    model.eval()

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.device = device
    app.state.mode = "lora" if use_lora else "base"

    print("Model loaded and ready.")
    yield


app = FastAPI(
    title="Anti-Sycophancy Model API",
    description="Local Qwen model with optional anti-sycophancy LoRA.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"name": "Anti-Sycophancy API", "docs": "/docs"}


@app.get("/health")
def health(request: Request):
    if not hasattr(request.app.state, "model"):
        return {"status": "starting"}

    return {
        "status": "ok",
        "device": request.app.state.device,
        "mode": request.app.state.mode,
        "base_model": BASE_MODEL,
    }


@app.post("/chat")
def chat(body: ChatRequest, request: Request):
    if body.message is not None and body.messages is not None:
        raise HTTPException(
            status_code=422,
            detail="Provide either message or messages, not both.",
        )

    if body.messages is not None:
        if not body.messages:
            raise HTTPException(status_code=422, detail="messages is empty.")

        messages = [m.model_dump() for m in body.messages]

        if not any(m["role"] == "system" for m in messages):
            messages.insert(
                0, {"role": "system", "content": SYSTEM_PROMPT}
            )

        if messages[-1]["role"] != "user":
            raise HTTPException(
                status_code=422,
                detail="The last message must have role user.",
            )
    else:
        if not body.message or not body.message.strip():
            raise HTTPException(
                status_code=422,
                detail="Please provide a non-empty message.",
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body.message.strip()},
        ]

    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[-1]

        generation = {
            "max_new_tokens": body.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": body.temperature > 0,
        }

        if body.temperature > 0:
            generation["temperature"] = body.temperature

        with INFERENCE_LOCK, torch.inference_mode():
            outputs = model.generate(**inputs, **generation)

        new_tokens = outputs[0, prompt_tokens:]
        answer = tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        return {
            "response": answer,
            "model": BASE_MODEL,
            "mode": request.app.state.mode,
            "usage": {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(len(new_tokens)),
            },
        }

    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=503,
            detail="GPU memory exhausted. Reduce the input or output length.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)