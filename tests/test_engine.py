# -*- coding: utf-8 -*-
"""CheckinEngine 签到引擎单元测试（防重复签到：冷却期 + 每日去重）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from checkin import CheckinEngine
from utils.storage import AttendanceRecorder


class FakeDetector:
    """假检测器：按顺序返回预设的人脸框"""
    def __init__(self, boxes):
        self.boxes = boxes
        self.backend = "fake"

    def detect(self, frame):
        return list(self.boxes)


class FakeEmbedder:
    """假特征提取器：始终返回预设的特征向量（模拟同一人反复出现在画面中）"""
    def __init__(self, embedding):
        self.embedding = embedding

    def extract(self, frame, box):
        return self.embedding


BOX = (10, 10, 100, 100)
VEC_A = np.zeros(128)
VEC_FAR = np.full(128, 5.0)   # 与 VEC_A 距离约 56.6，远超阈值


def make_engine(tmp_path, database, embedding=VEC_A, cooldown=60):
    recorder = AttendanceRecorder(tmp_path)
    engine = CheckinEngine(FakeDetector([BOX]), FakeEmbedder(embedding),
                           database, recorder, tolerance=0.55, cooldown=cooldown)
    return engine, recorder


def test_first_detection_checks_in(tmp_path):
    engine, recorder = make_engine(tmp_path, {"张三": VEC_A})
    detections, events = engine.process_frame(np.zeros((200, 200, 3), np.uint8))
    assert detections[0].name == "张三"
    assert detections[0].status == "new"
    assert any("签到成功" in e for e in events)
    assert recorder.load_checked_names() == {"张三"}
    assert len(engine.new_today) == 1


def test_cooldown_prevents_repeat_checkin(tmp_path):
    engine, recorder = make_engine(tmp_path, {"张三": VEC_A}, cooldown=60)
    engine.process_frame(np.zeros((200, 200, 3), np.uint8))          # 第一次：签到
    detections, events = engine.process_frame(np.zeros((200, 200, 3), np.uint8))  # 紧接着：冷却中
    assert detections[0].status == "cooldown"
    assert events == []
    lines = recorder.path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(lines) == 2   # 表头 + 1 条记录，没有重复写入


def test_after_cooldown_expires_status_is_checked_in(tmp_path):
    engine, recorder = make_engine(tmp_path, {"张三": VEC_A}, cooldown=0)
    engine.process_frame(np.zeros((200, 200, 3), np.uint8))
    detections, _ = engine.process_frame(np.zeros((200, 200, 3), np.uint8))
    assert detections[0].status == "checked_in"   # 冷却已过，但仍不重复签到
    lines = recorder.path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(lines) == 2


def test_already_checked_today_from_csv_is_skipped(tmp_path):
    # 模拟程序重启：CSV 里已有今天的签到记录
    recorder = AttendanceRecorder(tmp_path)
    recorder.record("张三")
    engine = CheckinEngine(FakeDetector([BOX]), FakeEmbedder(VEC_A),
                           {"张三": VEC_A}, AttendanceRecorder(tmp_path), 0.55, 60)
    detections, events = engine.process_frame(np.zeros((200, 200, 3), np.uint8))
    assert detections[0].status == "checked_in"
    assert events == []
    assert engine.new_today == []


def test_unknown_person_is_not_recorded(tmp_path):
    engine, recorder = make_engine(tmp_path, {"张三": VEC_A}, embedding=VEC_FAR)
    detections, events = engine.process_frame(np.zeros((200, 200, 3), np.uint8))
    assert detections[0].name is None
    assert detections[0].status == "unknown"
    assert recorder.load_checked_names() == set()


def test_empty_database_marks_unknown(tmp_path):
    engine, recorder = make_engine(tmp_path, {})
    detections, _ = engine.process_frame(np.zeros((200, 200, 3), np.uint8))
    assert detections[0].status == "unknown"
    assert recorder.load_checked_names() == set()
