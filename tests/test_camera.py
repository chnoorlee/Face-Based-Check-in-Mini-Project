# -*- coding: utf-8 -*-
"""摄像头健康检查单元测试（不依赖真实摄像头）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.camera import analyze_frame, diagnose_frame, warmup


class FakeCapture:
    """假摄像头：固定返回同一类画面"""
    def __init__(self, frame, fail_after=None):
        self.frame = frame
        self.reads = 0
        self.fail_after = fail_after

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        return True, self.frame.copy()


def blank_frame(value=51):
    return np.full((240, 320, 3), value, dtype=np.uint8)


def textured_frame():
    """带结构的画面：黑白棋盘格"""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(0, 240, 20):
        for x in range(0, 320, 20):
            if (x // 20 + y // 20) % 2:
                frame[y:y + 20, x:x + 20] = 200
    return frame


def test_warmup_waits_for_minimum_time():
    # 固定画面亮度不变，但必须等够 min_seconds 才允许判定稳定
    cap = FakeCapture(blank_frame())
    count, last = warmup(cap, min_seconds=0.15, max_seconds=1.0)
    assert count > 0 and last is not None
    assert cap.reads == count


def test_warmup_stops_when_camera_fails():
    cap = FakeCapture(blank_frame(), fail_after=3)
    count, last = warmup(cap, min_seconds=0, max_seconds=5)
    assert count == 3 and last is not None


def test_warmup_stops_at_max_seconds_with_unstable_brightness():
    class FlickerCapture:
        """亮度持续跳变：永远不稳定，必须在 max_seconds 截止"""
        def __init__(self):
            self.i = 0
        def read(self):
            self.i += 1
            frame = blank_frame(10 if self.i % 2 else 240)
            return True, frame
    count, _ = warmup(FlickerCapture(), min_seconds=0.05, max_seconds=0.2, tolerance=1.0)
    assert count > 0


def test_blank_frame_has_no_structure():
    stats = analyze_frame(blank_frame())
    assert stats["edge_ratio"] < 0.003
    assert stats["colorfulness"] < 1.5


def test_textured_frame_has_structure():
    stats = analyze_frame(textured_frame())
    assert stats["edge_ratio"] > 0.01


def test_diagnose_flags_covered_camera():
    """实测场景：隐私挡板关闭 → 纯灰画面，必须给出明确提示"""
    ok, message = diagnose_frame(blank_frame(51))
    assert ok is False
    assert "纯色" in message and "挡板" in message


def test_diagnose_flags_dark_frame():
    dark = textured_frame()
    dark[dark > 0] = 12          # 有结构但极暗
    ok, message = diagnose_frame(dark)
    assert ok is False and "过暗" in message


def test_diagnose_flags_overexposed_frame():
    bright = blank_frame(250)
    ok, message = diagnose_frame(bright)
    assert ok is False and ("过曝" in message or "纯色" in message)


def test_diagnose_accepts_normal_frame():
    ok, message = diagnose_frame(textured_frame())
    assert ok is True and "正常" in message


def test_analyze_handles_empty_input():
    stats = analyze_frame(None)
    assert stats["edge_ratio"] == 0.0
    assert diagnose_frame(None)[0] is False
