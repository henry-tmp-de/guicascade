"""一个极小的 OpenAI 兼容推理服务，跑在本机或服务器上。

## 为什么自己写

框架的 `Model` 层对接的是 OpenAI Chat Completions 协议。要让真实模型跑起来，
需要一个说这个协议的服务。可选方案有三条：

  1. **vLLM** —— 吞吐最好，但服务器上装不上（缺 python3.10-venv、网络中断），
     而且只有跑并发多环境时才需要它。
  2. **FastAPI + uvicorn** —— 要多装两个包。
  3. **标准库 http.server** —— 零新依赖。

这里选 3。理由很实际：**单任务顺序执行时，真正的开销是模型前向，不是 HTTP 框架。**
vLLM 的连续批处理在"一个模拟器、一步一次调用"的场景下发挥不出来。
等到要并行开 4~8 个模拟器时再上 vLLM，接口不用改（都是 OpenAI 协议）。

## 多模态怎么处理

GUI agent 每步要传一张截图。请求体里 `content` 是数组，图片走
`data:image/png;base64,...` 内联。这里把它解出来交给 processor。

## 用法

    python scripts/serve_models.py --model /path/to/qwen3-vl-2b --port 8000
    python scripts/serve_models.py --model /path/to/qwen3-vl-8b --port 8001 --device cuda:1

然后框架侧配置里写 `base_url: http://host:8000/v1` 即可。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_STATE: dict[str, Any] = {}


def _load(model_path: str, device: str, dtype: str) -> None:
    import torch
    from transformers import AutoProcessor

    # Qwen3-VL 的类名在不同 transformers 版本里有差异，按可用性依次尝试
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls

    print(f"[serve] 加载 processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"[serve] 加载模型: {model_path} -> {device} ({dtype})")
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, dtype)
    model = ModelCls.from_pretrained(model_path, **kwargs).to(device).eval()

    _STATE.update(processor=processor, model=model, device=device, torch=torch)
    print("[serve] 就绪")


def _to_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """把 OpenAI 格式的消息转成 chat template 能吃的形态。

    图片从 data URI 解成 PIL Image——processor 需要的是图像对象，不是 base64。
    """
    from PIL import Image

    out = []
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue

        parts = []
        for part in content or []:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                m = re.match(r"data:image/\w+;base64,(.*)", url, re.DOTALL)
                raw = base64.b64decode(m.group(1)) if m else b""
                parts.append({"type": "image", "image": Image.open(io.BytesIO(raw)).convert("RGB")})
        out.append({"role": msg["role"], "content": parts})
    return out


def _generate(body: dict[str, Any]) -> dict[str, Any]:
    torch = _STATE["torch"]
    processor = _STATE["processor"]
    model = _STATE["model"]
    device = _STATE["device"]

    messages = _to_messages(body)

    # 先把多模态内容喂给 processor 的聊天模板，再统一张量化
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [
        p["image"]
        for msg in messages
        if isinstance(msg.get("content"), list)
        for p in msg["content"]
        if p.get("type") == "image"
    ]

    inputs = processor(text=[text], images=images or None, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(body.get("max_tokens") or 2048),
            do_sample=float(body.get("temperature") or 0) > 0,
            temperature=max(float(body.get("temperature") or 0), 1e-5),
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "local"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # 默认每个请求打两行日志，跑实验时会刷屏；只在出错时打
        if args and str(args[1]).startswith(("4", "5")):
            print(f"[serve] {fmt % args}")

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        # 健康检查：框架侧可以先戳一下再发正式请求
        self._send(200, {"object": "list", "data": [{"id": "local", "object": "model"}]})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send(400, {"error": {"message": f"invalid JSON: {e}"}})
            return

        try:
            self._send(200, _generate(body))
        except Exception as e:  # noqa: BLE001 - 服务不该因为单次推理失败而挂掉
            import traceback

            traceback.print_exc()
            self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="auto", help="auto / bfloat16 / float16")
    args = ap.parse_args()

    _load(args.model, args.device, args.dtype)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] 监听 http://{args.host}:{args.port}/v1  (Ctrl-C 退出)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
