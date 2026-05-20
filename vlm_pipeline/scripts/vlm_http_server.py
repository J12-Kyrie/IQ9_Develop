#!/usr/bin/env python3
"""Lightweight HTTP wrapper around the IQ9 Qwen3-VL pipeline.

Exposes OpenAI-compatible /v1/chat/completions endpoint.
Internally manages VEG server + genie-t2t-run lifecycle.

Usage:
    python3 vlm_http_server.py --bundle /mnt/workspace/bundle_iq9 --port 8000
"""
import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Lock

# ── globals ──
BUNDLE_DIR: Path = None
VEG_PID: int = 0
GENIE_PROC: subprocess.Popen = None
GENIE_LOCK = Lock()
VEG_SOCK = "/tmp/veg_server.sock"
VEG_EMBEDS = "/tmp/veg_embeds.raw"


def cleanup(signum=None, frame=None):
    global VEG_PID, GENIE_PROC
    if GENIE_PROC and GENIE_PROC.poll() is None:
        GENIE_PROC.terminate()
        try:
            GENIE_PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            GENIE_PROC.kill()
    if VEG_PID:
        try:
            os.kill(VEG_PID, signal.SIGTERM)
        except OSError:
            pass
    # clean up sockets/files
    for f in (VEG_SOCK, VEG_EMBEDS):
        try:
            os.unlink(f)
        except OSError:
            pass
    if signum is not None:
        sys.exit(0)


def start_veg_server():
    global VEG_PID
    veg_bin = BUNDLE_DIR / "bin" / "veg-server-combined"
    veg_model = BUNDLE_DIR / "models" / "veg_qwen3.bin"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BUNDLE_DIR / 'lib'}:{BUNDLE_DIR / 'lib' / 'dsp'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["ADSP_LIBRARY_PATH"] = str(BUNDLE_DIR / "lib" / "dsp")

    log_f = open(BUNDLE_DIR / "logs" / "veg_server.log", "w")
    proc = subprocess.Popen(
        [str(veg_bin),
         "--backend", str(BUNDLE_DIR / "lib" / "libQnnHtp.so"),
         "--system_library", str(BUNDLE_DIR / "lib" / "libQnnSystem.so"),
         "--model", str(veg_model)],
        env=env, stdout=log_f, stderr=log_f,
    )
    VEG_PID = proc.pid
    print(f"[vlm-server] VEG server PID={VEG_PID}")

    # wait for socket
    for i in range(30):
        if os.path.exists(VEG_SOCK):
            print(f"[vlm-server] VEG server ready ({i+1}s)")
            return
        time.sleep(1)
    raise RuntimeError(f"VEG server failed to start (socket {VEG_SOCK} not found)")


def start_genie():
    global GENIE_PROC
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BUNDLE_DIR / 'lib'}:{BUNDLE_DIR / 'lib' / 'dsp'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["ADSP_LIBRARY_PATH"] = str(BUNDLE_DIR / "lib" / "dsp")

    log_f = open(BUNDLE_DIR / "logs" / "genie_server.log", "w")
    GENIE_PROC = subprocess.Popen(
        [str(BUNDLE_DIR / "bin" / "genie-t2t-run"),
         "-c", "qwen3_iq9.json",
         "--vocab", "tokenizer.json",
         "-t", "models/embedding_int8_lut.bin,uint8,0.0017348345136269927,-114",
         "--vision_embed", VEG_EMBEDS,
         "--log", "warn"],
        cwd=str(BUNDLE_DIR),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log_f,
    )
    print(f"[vlm-server] genie-t2t-run PID={GENIE_PROC.pid}")
    # wait for "ready" signal or first line
    # genie outputs a prompt or ready marker on stdout


def process_image_via_veg(jpeg_bytes: bytes):
    """Write JPEG to temp file and call VEG to produce embeddings."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp", delete=False) as f:
        f.write(jpeg_bytes)
        tmp_path = f.name
    try:
        # VEG server processes images via a client tool or direct socket
        # For now, use the veg-server-combined's CLI mode if available
        # Otherwise, we need to use the raw socket protocol
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{BUNDLE_DIR / 'lib'}:{BUNDLE_DIR / 'lib' / 'dsp'}:{env.get('LD_LIBRARY_PATH', '')}"
        env["ADSP_LIBRARY_PATH"] = str(BUNDLE_DIR / "lib" / "dsp")

        # Check if there's a veg client tool
        veg_client = BUNDLE_DIR / "bin" / "veg-client"
        if veg_client.exists():
            result = subprocess.run(
                [str(veg_client), "--image", tmp_path, "--output", VEG_EMBEDS],
                env=env, capture_output=True, timeout=30,
            )
            return result.returncode == 0
        else:
            # VEG server uses Unix socket protocol — write image to a known path
            # and signal VEG to process it
            shutil.copy(tmp_path, "/tmp/veg_input.jpg")
            # TODO: implement VEG socket protocol
            return False
    finally:
        os.unlink(tmp_path)


def call_genie(prompt: str, timeout: int = 60) -> str:
    """Send prompt to genie-t2t-run via stdin, read response from stdout."""
    with GENIE_LOCK:
        GENIE_PROC.stdin.write((prompt + "\n").encode())
        GENIE_PROC.stdin.flush()

        # Read response until EOF or timeout marker
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            line = GENIE_PROC.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip("\n")
            lines.append(decoded)
            # genie typically outputs a completion marker
            if decoded.strip() == "" and lines:
                break
        return "\n".join(lines)


class VLMHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404, "Not Found")
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Extract image and prompt from messages
        messages = req.get("messages", [])
        prompt = ""
        image_b64 = None
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        prompt = part.get("text", "")
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            image_b64 = url.split(",", 1)[1]
            elif isinstance(content, str):
                prompt = content

        t0 = time.time()

        # Process image through VEG if available
        if image_b64:
            jpeg_bytes = base64.b64decode(image_b64)
            # TODO: send to VEG server for embedding generation

        # Call genie for text generation
        with GENIE_LOCK:
            response_text = call_genie(prompt, timeout=req.get("timeout", 60))

        latency_ms = (time.time() - t0) * 1000

        resp = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", "qwen3-vl-4b"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        resp_bytes = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            resp = json.dumps({"status": "ok"}).encode()
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp)
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        print(f"[vlm-server] {args[0]}")


def main():
    global BUNDLE_DIR

    parser = argparse.ArgumentParser(description="VLM HTTP Server")
    parser.add_argument("--bundle", required=True, help="Path to bundle_iq9 directory")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--skip-veg", action="store_true", help="Skip VEG server startup (already running)")
    parser.add_argument("--skip-genie", action="store_true", help="Skip genie startup (already running)")
    args = parser.parse_args()

    BUNDLE_DIR = Path(args.bundle).resolve()
    if not BUNDLE_DIR.exists():
        print(f"ERROR: bundle dir not found: {BUNDLE_DIR}")
        sys.exit(1)

    # Ensure log dir
    (BUNDLE_DIR / "logs").mkdir(exist_ok=True)

    # Setup signal handlers
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    # Start VLM pipeline
    if not args.skip_veg:
        start_veg_server()
    if not args.skip_genie:
        start_genie()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), VLMHandler)
    print(f"[vlm-server] Listening on 0.0.0.0:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
