#!/usr/bin/env python3
"""HTTP front of `nntr_discretion --serve --speech`: the speech models of another program's pipeline.

ALICE (ambient-agent-alice) keeps its own endpointing and speaker clustering and calls this server for the models:

  GET  /health                 {"status": "ready", "asr_secs": [...], "queue": n, ...}
  POST /asr      body: PCM16   {"text", "words": [[start_ms, end_ms, word], ...], "pieces", "ms"}
  POST /embed    body: PCM16   {"vector": [256 floats], "ms"}
  POST /segment  body: PCM16   {"frame_ms", "windows": [{"offset": samples, "classes": "0011..."}], "ms"}

Bodies are 16 kHz mono little-endian PCM16. One resident worker answers one request at a time (each request already uses
all its cores); it restarts if it dies. With --token, requests need `Authorization: Bearer TOKEN`. Stdlib only, so it
runs on the device next to the aarch64 binary:

    python3 speech_server.py --bin ./nntr_discretion --model ~/workspace/new/disc/models/Qwen3-ASR-1.7B \\
        --sd-weights speaker_diarization.bin --host 0.0.0.0 --port 8030
"""

import argparse
import hmac
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 120 * 16000 * 2  # two minutes of audio
OPS = {"/asr": "ASR", "/embed": "EMBED", "/segment": "SEGMENT"}


class Worker:
    """One `nntr_discretion --serve --speech` process; the models load once."""

    def __init__(self, binary, model, sd_weights, language, asr_secs, threads):
        self.command = [str(binary), str(model), "--serve", "--speech", "--sd-weights", str(sd_weights),
                        "--language", language, "--asr-sec", asr_secs]
        if threads:
            self.command += ["--asr-threads", str(threads)]
        self.env = dict(os.environ, LD_LIBRARY_PATH=str(Path(binary).resolve().parent))
        self.proc = None
        self.info = {}
        self.lock = threading.Lock()
        self.waiting = 0
        self.count = threading.Lock()

    def _start(self):
        if self.proc is not None:  # the previous process died; release its pipes
            for pipe in (self.proc.stdin, self.proc.stdout):
                pipe.close()
            self.proc.wait()
        self.proc = subprocess.Popen(self.command, env=self.env, text=True, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        line = self.proc.stdout.readline().strip()
        if not line.startswith("READY"):
            self.proc.kill()
            self.proc = None
            raise RuntimeError("nntr_discretion did not become ready")
        self.info = json.loads(line[5:].strip() or "{}")

    def ensure_started(self):
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()

    def run(self, op, pcm, work_dir):
        with self.count:
            self.waiting += 1
        try:
            with self.lock:
                if self.proc is None or self.proc.poll() is not None:
                    self._start()  # restart after a crash; the models reload once
                with tempfile.TemporaryDirectory(dir=work_dir) as folder:
                    src, out = Path(folder) / "in.pcm", Path(folder) / "out.json"
                    src.write_bytes(pcm)
                    self.proc.stdin.write(f"{op}\t{src}\t{out}\n")
                    self.proc.stdin.flush()
                    reply = self.proc.stdout.readline().strip()
                    if not reply:
                        raise RuntimeError("nntr_discretion exited while answering")
                    if reply != "OK":
                        raise ValueError(reply[:300])
                    return out.read_bytes()
        finally:
            with self.count:
                self.waiting -= 1


def make_handler(worker, work_dir, token):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _allowed(self):
            if not token:
                return True
            given = self.headers.get("Authorization", "")
            return hmac.compare_digest(given.encode(), f"Bearer {token}".encode())

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"detail": "not found"})
            if not self._allowed():
                return self._send(401, {"detail": "token required"})
            alive = worker.proc is not None and worker.proc.poll() is None
            self._send(200, {"status": "ready" if alive else "starting", "queue": worker.waiting, **worker.info})

        def do_POST(self):
            op = OPS.get(self.path)
            if op is None:
                return self._send(404, {"detail": "not found"})
            if not self._allowed():
                return self._send(401, {"detail": "token required"})
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > MAX_BODY_BYTES or length % 2:
                return self._send(413 if length > MAX_BODY_BYTES else 400,
                                  {"detail": "body must be 16 kHz mono PCM16, up to two minutes"})
            pcm = self.rfile.read(length)
            started = time.monotonic()
            try:
                body = worker.run(op, pcm, work_dir)
            except ValueError as exc:
                return self._send(422, {"detail": str(exc)})
            except Exception as exc:  # the worker died or could not start
                return self._send(503, {"detail": str(exc)[:300]})
            self.log_message("%s %d bytes %.0f ms", op, length, (time.monotonic() - started) * 1000)
            self._send(200, body)

        def log_message(self, fmt, *args):
            print("%s %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bin", required=True, help="path to nntr_discretion")
    parser.add_argument("--model", required=True, help="Qwen3-ASR model directory")
    parser.add_argument("--sd-weights", required=True, help="speaker_diarization.bin")
    parser.add_argument("--language", default="Korean")
    parser.add_argument("--asr-sec", default="8,12.5",
                        help="ASR model graph lengths in seconds; a segment uses the shortest that fits")
    parser.add_argument("--threads", type=int, default=0, help="ASR threads (0 = all cores)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--work-dir", default="/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
    parser.add_argument("--token", default=os.environ.get("NNTR_SPEECH_TOKEN", ""),
                        help="require Authorization: Bearer TOKEN (default: $NNTR_SPEECH_TOKEN)")
    args = parser.parse_args()
    worker = Worker(args.bin, args.model, args.sd_weights, args.language, args.asr_sec, args.threads)
    worker.ensure_started()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(worker, args.work_dir, args.token))
    print(f"nntr speech server on {args.host}:{args.port} {json.dumps(worker.info)}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
