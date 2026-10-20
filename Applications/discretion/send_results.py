#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   send_results.py
# @brief  Follow a JSON Lines file written by nntr_discretion (--jsonl) and deliver
#         each object through a transport, with retries, a dead-letter file and a
#         resumable offset. The transport is pluggable; only a local file transport
#         exists so far (no network).
"""
Delivery pipeline

  nntr_discretion --jsonl FILE   (writes FILE; the local record is always kept)
        |
  send_results.py FILE           follows FILE from the saved offset
        |  producer thread: reads complete lines only
        |  queue (bounded): a slow or down server never blocks the recognizer
        |  worker thread: parse -> transport.send() with retries and backoff
        v
  transport (file:PATH now; HTTP/WebSocket/TCP later)

Files next to FILE:
  FILE.sent   byte offset of the last line that was delivered or dead-lettered
  FILE.dead   objects that could not be delivered (with the reason)

Delivery is at-least-once: a crash between send() and the offset update repeats
the last object. The receiver can drop duplicates with (session, seq).
"""

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time


class FileTransport:
    """Stand-in transport: appends each object to a local file."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def send(self, obj):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class Delivery:
    """Sends one object with bounded retries; gives up into the dead letter."""

    def __init__(self, transport, retries=5, backoff=0.5, sleep=time.sleep,
                 dead_letter=None):
        self.transport = transport
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.dead_letter = dead_letter  # callable(obj_or_raw, reason) or None
        self.delivered = 0
        self.failed = 0

    def deliver(self, obj):
        last = None
        for attempt in range(self.retries + 1):
            try:
                self.transport.send(obj)
                self.delivered += 1
                return True
            except Exception as e:  # any transport error: retry, then give up
                last = e
                if attempt < self.retries:
                    self.sleep(self.backoff * (2 ** attempt))
        self.failed += 1
        if self.dead_letter:
            self.dead_letter(obj, "send failed after %d attempts: %s"
                             % (self.retries + 1, last))
        return False


class DeadLetter:
    """Appends undeliverable items, with the reason, to a JSON Lines file."""

    def __init__(self, path):
        self.path = path

    def __call__(self, item, reason):
        record = {"reason": reason, "item": item}
        if isinstance(item, bytes):
            record["item"] = item.decode("utf-8", errors="replace")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_offset(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return int(f.read().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0


def save_offset(state_path, offset):
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(offset))
    os.replace(tmp, state_path)  # atomic: a crash never leaves a half-written offset


def read_complete_lines(path, offset):
    """Return [(raw_line_bytes, offset_after_line), ...] for complete lines only."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return []
    end = data.rfind(b"\n")
    if end < 0:
        return []  # the writer has not finished a line yet
    out = []
    pos = offset
    for raw in data[:end + 1].splitlines(keepends=True):
        pos += len(raw)
        out.append((raw.rstrip(b"\r\n"), pos))
    return out


def process(raw, delivery, dead_letter):
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        dead_letter(raw, "invalid JSON: %s" % e)
        return
    delivery.deliver(obj)


def run(jsonl, transport, state_path, dead_path, retries, backoff,
        poll, once, stop_event=None, sleep=time.sleep):
    dead_letter = DeadLetter(dead_path)
    delivery = Delivery(transport, retries, backoff, sleep=sleep,
                        dead_letter=dead_letter)
    work = queue.Queue(maxsize=1024)

    def worker():
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            raw, end_offset = item
            process(raw, delivery, dead_letter)
            save_offset(state_path, end_offset)
            work.task_done()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    offset = load_offset(state_path)
    try:
        while True:
            lines = read_complete_lines(jsonl, offset)
            for raw, end_offset in lines:
                work.put((raw, end_offset))  # blocks only when 1024 items wait
                offset = end_offset
            if stop_event is not None and stop_event.is_set():
                break  # stop requested: what was read so far is still drained
            if once and not lines:
                break
            if not lines:
                time.sleep(poll)
    except KeyboardInterrupt:
        pass
    work.join()  # drain: everything read so far is delivered or dead-lettered
    work.put(None)
    thread.join()
    transport.close()
    return delivery


def parse_transport(spec):
    kind, _, arg = spec.partition(":")
    if kind == "file" and arg:
        return FileTransport(arg)
    raise SystemExit("unknown transport %r (use file:PATH)" % spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("jsonl", help="JSON Lines file written by nntr_discretion --jsonl")
    ap.add_argument("--transport", default=None,
                    help="file:PATH (default: file:FILE.out.jsonl)")
    ap.add_argument("--state", default=None, help="offset file (default: FILE.sent)")
    ap.add_argument("--dead-letter", default=None, help="default: FILE.dead")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--backoff", type=float, default=0.5,
                    help="first retry delay in seconds, doubled each attempt")
    ap.add_argument("--poll", type=float, default=0.2, help="seconds between reads")
    ap.add_argument("--once", action="store_true",
                    help="deliver what is in the file now and exit")
    args = ap.parse_args(argv)

    transport_spec = args.transport or ("file:" + args.jsonl + ".out.jsonl")
    state = args.state or (args.jsonl + ".sent")
    dead = args.dead_letter or (args.jsonl + ".dead")
    transport = parse_transport(transport_spec)

    # Explicit handlers: a background job started from a non-interactive shell
    # has SIGINT ignored, so Python's default KeyboardInterrupt would never come.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stop.set())
    delivery = run(args.jsonl, transport, state, dead, args.retries,
                   args.backoff, args.poll, args.once, stop_event=stop)
    print("delivered %d, failed %d (dead letter: %s)"
          % (delivery.delivered, delivery.failed, dead), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
