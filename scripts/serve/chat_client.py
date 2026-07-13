#!/usr/bin/env python3
"""多轮对话客户端 — 连接 mini-vllm HTTP 服务。

复用 chat.ChatUI 做界面展示，通过 SSE 流式接收 token。
每轮将完整对话历史发给服务端，服务端负责 chat template 拼接。

Usage:
    python scripts/chat_client.py
    python scripts/chat_client.py --url http://localhost:8000 --system "你是一个助手"
    python scripts/chat_client.py --no-stream   # 关闭流式，等全量结果
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

try:
    import readline
except ImportError:
    readline = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from chat.ui import ChatUI

DEFAULT_URL = "http://localhost:8000"


def stream_chat(
    base_url: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> tuple[str, str | None]:
    """Send messages to the server via SSE, print tokens as they arrive.

    Returns (full_text, finish_reason).
    """
    payload = {
        "model": "mini-vllm",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "stream": True,
    }
    tokens: list[str] = []
    finish_reason = None

    with requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=300,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if line == "data: [DONE]":
                break
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            choice = chunk["choices"][0]
            delta = choice["delta"].get("content") or ""
            if delta:
                tokens.append(delta)
                print(delta, end="", flush=True)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    print()  # newline after streaming
    return "".join(tokens), finish_reason


def no_stream_chat(
    base_url: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
) -> tuple[str, str | None]:
    """Non-streaming request; returns full response at once."""
    payload = {
        "model": "mini-vllm",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "stream": False,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return choice["message"]["content"], choice.get("finish_reason")


def check_server(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="mini-vllm 多轮对话客户端")
    parser.add_argument("--url", default=DEFAULT_URL, help="服务地址")
    parser.add_argument("--system", default=None, help="系统提示词")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    args = parser.parse_args()

    ui = ChatUI()

    if not check_server(args.url):
        ui.print_error(f"无法连接服务: {args.url}，请先启动 scripts/start_server.py")
        sys.exit(1)

    ui.print_header(model_name=args.url)

    messages: list[dict] = []
    enable_thinking = True
    turn = 1

    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user_text = ui.input_prompt()
        except (EOFError, KeyboardInterrupt):
            break

        user_text = user_text.strip()
        if not user_text:
            continue

        # Commands
        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/clear":
            system = [m for m in messages if m["role"] == "system"]
            messages = system
            turn = 1
            ui.print_info("对话历史已清空")
            continue
        if user_text == "/think on":
            enable_thinking = True
            ui.print_info("思考模式已开启")
            continue
        if user_text == "/think off":
            enable_thinking = False
            ui.print_info("思考模式已关闭")
            continue

        messages.append({"role": "user", "content": user_text})
        ui.print_user(user_text)

        t0 = time.perf_counter()
        try:
            if args.no_stream:
                full_text, finish_reason = no_stream_chat(
                    args.url, messages, args.max_tokens, args.temperature, enable_thinking
                )
                ui.print_assistant(full_text, turn, elapsed_ms=(time.perf_counter() - t0) * 1000, show_cot=enable_thinking)
            else:
                # print header before streaming starts
                ui.console.print(f"[bold green]Assistant (Turn {turn})[/bold green]")
                full_text, finish_reason = stream_chat(
                    args.url, messages, args.max_tokens, args.temperature, enable_thinking
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                ui.console.print(f"[dim]({elapsed_ms:.0f} ms, finish={finish_reason})[/dim]\n")
        except requests.HTTPError as e:
            ui.print_error(f"请求失败: {e}")
            messages.pop()  # rollback user message
            continue
        except Exception as e:
            ui.print_error(f"错误: {e}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": full_text})
        turn += 1

    ui.print_info("再见！")


if __name__ == "__main__":
    main()
