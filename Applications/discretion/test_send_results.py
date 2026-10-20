#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   test_send_results.py
# @brief  Unit tests for send_results.py (delivery order, retries, dead letter,
#         resume from the saved offset, partial lines, bad JSON).
"""Run with: python3 -m unittest -v test_send_results.py"""

import json
import os
import tempfile
import unittest

import send_results as sr


class RecordingTransport:
    """Records delivered objects; fails the attempts listed in fail_plan."""

    def __init__(self, fail_plan=None):
        self.sent = []
        self.attempts = {}  # seq -> number of send() calls
        self.fail_plan = fail_plan or {}  # seq -> number of initial failures
        self.closed = False

    def send(self, obj):
        seq = obj["seq"]
        n = self.attempts.get(seq, 0)
        self.attempts[seq] = n + 1
        if n < self.fail_plan.get(seq, 0):
            raise ConnectionError("simulated outage")
        self.sent.append(obj)

    def close(self):
        self.closed = True


def obj(seq, text=None):
    return {"session": "s1", "seq": seq, "start": float(seq), "end": seq + 1.0,
            "speaker": "SPEAKER_00", "lag": 1.0, "asr_lag": 0.5,
            "text": text or "line %d" % seq}


class SendResultsTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.jsonl = os.path.join(self.dir.name, "run.jsonl")
        self.state = self.jsonl + ".sent"
        self.dead = self.jsonl + ".dead"

    def tearDown(self):
        self.dir.cleanup()

    def write(self, objs, mode="w"):
        with open(self.jsonl, mode, encoding="utf-8") as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

    def run_once(self, transport, retries=2):
        return sr.run(self.jsonl, transport, self.state, self.dead,
                      retries=retries, backoff=0.0, poll=0.0, once=True,
                      sleep=lambda s: None)

    def saved_offset(self):
        with open(self.state, encoding="utf-8") as f:
            return int(f.read())

    def dead_items(self):
        if not os.path.exists(self.dead):
            return []
        with open(self.dead, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def test_delivers_all_in_order_and_saves_offset(self):
        self.write([obj(i) for i in range(5)])
        t = RecordingTransport()
        d = self.run_once(t)
        self.assertEqual([o["seq"] for o in t.sent], [0, 1, 2, 3, 4])
        self.assertEqual(d.delivered, 5)
        self.assertEqual(d.failed, 0)
        self.assertEqual(self.saved_offset(), os.path.getsize(self.jsonl))
        self.assertTrue(t.closed)

    def test_retries_then_succeeds(self):
        self.write([obj(i) for i in range(4)])
        t = RecordingTransport(fail_plan={2: 2})  # fails twice, third try works
        d = self.run_once(t, retries=2)
        self.assertEqual([o["seq"] for o in t.sent], [0, 1, 2, 3])
        self.assertEqual(t.attempts[2], 3)
        self.assertEqual(d.failed, 0)
        self.assertEqual(self.dead_items(), [])

    def test_dead_letter_after_retries_keeps_going(self):
        self.write([obj(i) for i in range(4)])
        t = RecordingTransport(fail_plan={1: 99})  # never succeeds
        d = self.run_once(t, retries=2)
        self.assertEqual([o["seq"] for o in t.sent], [0, 2, 3])
        self.assertEqual(t.attempts[1], 3)  # 1 + 2 retries
        self.assertEqual(d.failed, 1)
        dead = self.dead_items()
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["item"]["seq"], 1)
        self.assertIn("send failed", dead[0]["reason"])
        # the offset moved past the dead-lettered line, so it is not retried later
        self.assertEqual(self.saved_offset(), os.path.getsize(self.jsonl))

    def test_resume_from_saved_offset_has_no_duplicates(self):
        self.write([obj(0), obj(1), obj(2)])
        t1 = RecordingTransport()
        self.run_once(t1)
        self.write([obj(3), obj(4)], mode="a")  # more output arrives later
        t2 = RecordingTransport()
        self.run_once(t2)
        self.assertEqual([o["seq"] for o in t1.sent], [0, 1, 2])
        self.assertEqual([o["seq"] for o in t2.sent], [3, 4])

    def test_partial_line_is_not_read_until_complete(self):
        line = json.dumps(obj(0)) + "\n"
        with open(self.jsonl, "w", encoding="utf-8") as f:
            f.write(line)
            f.write(json.dumps(obj(1))[:10])  # writer is mid-line
        t = RecordingTransport()
        self.run_once(t)
        self.assertEqual([o["seq"] for o in t.sent], [0])
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj(1))[10:] + "\n")
        t2 = RecordingTransport()
        self.run_once(t2)
        self.assertEqual([o["seq"] for o in t2.sent], [1])

    def test_bad_json_goes_to_dead_letter(self):
        with open(self.jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj(0)) + "\n")
            f.write("{not json}\n")
            f.write(json.dumps(obj(2)) + "\n")
        t = RecordingTransport()
        d = self.run_once(t)
        self.assertEqual([o["seq"] for o in t.sent], [0, 2])
        dead = self.dead_items()
        self.assertEqual(len(dead), 1)
        self.assertIn("invalid JSON", dead[0]["reason"])
        self.assertEqual(d.failed, 0)  # bad lines are not counted as send failures

    def test_korean_text_is_preserved(self):
        self.write([obj(0, text="그리스하면 페르시아를 떠올리는 거 같아요.")])
        t = RecordingTransport()
        self.run_once(t)
        self.assertEqual(t.sent[0]["text"], "그리스하면 페르시아를 떠올리는 거 같아요.")

    def test_file_transport_writes_json_lines(self):
        out = os.path.join(self.dir.name, "out.jsonl")
        self.write([obj(0), obj(1)])
        sr.run(self.jsonl, sr.FileTransport(out), self.state, self.dead,
               retries=0, backoff=0.0, poll=0.0, once=True, sleep=lambda s: None)
        with open(out, encoding="utf-8") as f:
            self.assertEqual([json.loads(line)["seq"] for line in f], [0, 1])


if __name__ == "__main__":
    unittest.main()
