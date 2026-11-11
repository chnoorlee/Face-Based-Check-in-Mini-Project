# -*- coding: utf-8 -*-
"""FeatureDatabase / AttendanceRecorder 单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.storage import AttendanceRecorder, FeatureDatabase, is_late


def _make_image():
    return np.zeros((64, 64, 3), dtype=np.uint8)


class TestFeatureDatabase:
    def test_save_and_load_roundtrip(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        db.save("张三", np.ones(128), _make_image())
        loaded = db.load_all()
        assert set(loaded) == {"张三"}
        np.testing.assert_array_equal(loaded["张三"], np.ones(128))
        # 人脸照片也应保存成功
        assert (db.faces_dir / "张三.jpg").exists()

    def test_names_and_exists(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        assert db.names() == []
        db.save("李四", np.zeros(128), _make_image())
        assert db.names() == ["李四"]
        assert db.exists("李四")
        assert not db.exists("王五")

    def test_illegal_characters_in_name_are_sanitized(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        db.save('张/三:*?', np.zeros(128), _make_image())
        assert len(db.names()) == 1
        assert "/" not in db.names()[0] and ":" not in db.names()[0]

    def test_leading_formula_chars_are_sanitized(self, tmp_path):
        # 防止 CSV 公式注入：以 = 开头的姓名应被改写
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        db.save("=cmd", np.zeros(128), _make_image())
        assert db.names()[0] == "_cmd"

    def test_overwrite_same_name(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        db.save("张三", np.ones(128), _make_image())
        db.save("张三", np.zeros(128), _make_image())
        assert len(db.names()) == 1
        np.testing.assert_array_equal(db.load_all()["张三"], np.zeros(128))

    def test_delete_removes_all_files(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        db.save("张三", np.ones(128), _make_image())
        assert db.delete("张三")
        assert db.names() == []
        assert not (db.faces_dir / "张三.jpg").exists()

    def test_delete_unknown_returns_false(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        assert not db.delete("不存在的人")

    def test_details_include_registered_at(self, tmp_path):
        db = FeatureDatabase(tmp_path / "emb", tmp_path / "faces")
        db.save("张三", np.ones(128), _make_image())
        infos = db.details()
        assert len(infos) == 1
        assert infos[0]["name"] == "张三"
        assert "registered_at" in infos[0] and "20" in infos[0]["registered_at"]


class TestAttendanceRecorder:
    def test_record_writes_header_once(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        rec.record("张三")
        rec.record("李四")
        lines = rec.path.read_text(encoding="utf-8-sig").strip().splitlines()
        assert lines[0] == "学生姓名,签到时间,是否签到成功"
        assert len(lines) == 3   # 表头 + 两条记录

    def test_record_row_content(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        row = rec.record("张三")
        assert row["学生姓名"] == "张三"
        assert row["是否签到成功"] == "是"

    def test_load_checked_names_only_counts_success(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        rec.record("张三")
        rec.record("李四", success=False)
        assert rec.load_checked_names() == {"张三"}

    def test_new_instance_keeps_history(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        rec.record("张三")
        rec2 = AttendanceRecorder(tmp_path)   # 模拟程序重启后重新打开
        assert rec2.load_checked_names() == {"张三"}

    def test_empty_file_returns_empty_set(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        assert rec.load_checked_names() == set()

    def test_corrupted_csv_is_tolerated(self, tmp_path):
        # 模拟 CSV 被写坏（非 UTF-8 字节）——按空记录处理，不崩溃
        rec = AttendanceRecorder(tmp_path)
        rec.path.write_bytes(b"\xff\xfe\x00garbage\x9c")
        assert rec.load_checked_names() == set()

    def test_malformed_rows_are_skipped(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        rec.path.write_text("学生姓名,签到时间,是否签到成功\n只有一列\n张三,2026-11-11 08:00:00,是\n",
                            encoding="utf-8-sig")
        assert rec.load_checked_names() == {"张三"}

    def test_load_records_maps_name_to_time(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        rec.record("张三")
        records = rec.load_records()
        assert set(records) == {"张三"}
        assert records["张三"].startswith("20")

    def test_write_summary_marks_present_and_absent(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        rec.record("张三")
        path = rec.write_summary(["张三", "李四"], {"张三"})
        lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
        assert lines[0] == "学生姓名,状态,签到时间"
        assert lines[1].startswith("张三,已到,")
        assert lines[2] == "李四,缺勤,"

    def test_write_summary_without_any_checkin(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        path = rec.write_summary(["张三", "李四"], set())
        lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
        assert len(lines) == 3
        assert all("缺勤" in line for line in lines[1:])

    def test_write_summary_marks_leave(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        path = rec.write_summary(["张三", "李四"], {"张三"}, leave_names=["李四"])
        lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
        assert lines[2] == "李四,请假,"

    def test_write_summary_marks_late(self, tmp_path):
        from datetime import time
        rec = AttendanceRecorder(tmp_path)
        rec.path.write_text(
            "学生姓名,签到时间,是否签到成功\n"
            "张三,2026-11-11 08:05:00,是\n"
            "李四,2026-11-11 08:30:00,是\n",
            encoding="utf-8-sig")
        path = rec.write_summary(["张三", "李四"], {"张三", "李四"},
                                 late_after=time(8, 10))
        text = path.read_text(encoding="utf-8-sig")
        assert "张三,已到,2026-11-11 08:05:00" in text
        assert "李四,迟到,2026-11-11 08:30:00" in text

    def test_is_late_helper(self):
        from datetime import time
        assert not is_late("2026-11-11 08:05:00", time(8, 10))
        assert is_late("2026-11-11 08:15:00", time(8, 10))
        assert not is_late("2026-11-11 08:15:00", None)
        assert not is_late("", time(8, 10))
        assert not is_late("乱码时间", time(8, 10))


class TestSession:
    def test_session_in_filename(self, tmp_path):
        rec = AttendanceRecorder(tmp_path, session="第1节")
        assert rec.path.name == "attendance_" + rec.date + "_第1节.csv"
        assert rec.report_path().name == "attendance_" + rec.date + "_第1节_report.csv"

    def test_no_session_keeps_legacy_name(self, tmp_path):
        rec = AttendanceRecorder(tmp_path)
        assert rec.path.name == "attendance_" + rec.date + ".csv"

    def test_two_sessions_are_independent(self, tmp_path):
        s1 = AttendanceRecorder(tmp_path, session="第1节")
        s2 = AttendanceRecorder(tmp_path, session="第2节")
        s1.record("张三")
        assert s1.load_checked_names() == {"张三"}
        assert s2.load_checked_names() == set()   # 第2节不受第1节影响
        s2.record("张三")
        assert s2.load_checked_names() == {"张三"}
