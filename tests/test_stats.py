# -*- coding: utf-8 -*-
"""stats 出勤统计报表单元测试"""
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import stats
from utils.storage import AttendanceRecorder


def make_day(tmp_path, date, names, session=None):
    """在临时目录造一天的签到记录"""
    rec = AttendanceRecorder(tmp_path, date=date, session=session)
    for n in names:
        rec.record(n)
    return rec


def test_discover_sessions_groups_report_and_detail(tmp_path):
    rec = AttendanceRecorder(tmp_path, date="2026-11-11")
    rec.record("张三")
    rec.write_summary(["张三"], {"张三"})
    sessions = stats.discover_sessions(tmp_path)
    assert "2026-11-11" in sessions
    assert sessions["2026-11-11"]["detail"] is not None
    assert sessions["2026-11-11"]["report"] is not None


def test_discover_sessions_distinguishes_sessions(tmp_path):
    make_day(tmp_path, "2026-11-11", ["张三"], session="第1节")
    make_day(tmp_path, "2026-11-11", ["李四"], session="第2节")
    keys = sorted(stats.discover_sessions(tmp_path))
    assert keys == ["2026-11-11_第1节", "2026-11-11_第2节"]


def test_summarize_counts_attendance(tmp_path):
    make_day(tmp_path, "2026-11-11", ["张三"])
    make_day(tmp_path, "2026-11-12", ["张三", "李四"])
    rows, keys = stats.summarize(tmp_path, late_after=time(8, 10))
    assert len(keys) == 2
    table = {r["姓名"]: r for r in rows}
    assert table["张三"]["应到"] == 2 and table["张三"]["实到"] == 2
    assert table["张三"]["出勤率"] == "100.0%"
    assert table["李四"]["实到"] == 1 and table["李四"]["缺勤"] == 1
    assert table["李四"]["出勤率"] == "50.0%"


def test_summarize_uses_report_for_leave(tmp_path):
    rec = make_day(tmp_path, "2026-11-11", ["张三"])
    rec.write_summary(["张三", "李四"], {"张三"}, leave_names=["李四"])
    rows, _ = stats.summarize(tmp_path)
    table = {r["姓名"]: r for r in rows}
    assert table["李四"]["请假"] == 1
    assert table["李四"]["缺勤"] == 0


def test_summarize_filters_by_date_range(tmp_path):
    make_day(tmp_path, "2026-09-01", ["张三"])
    make_day(tmp_path, "2026-11-11", ["张三"])
    rows, keys = stats.summarize(tmp_path, date_from="2026-11-01")
    assert keys == ["2026-11-11"]
    assert rows[0]["应到"] == 1


def test_summarize_restricted_to_roster(tmp_path):
    make_day(tmp_path, "2026-11-11", ["张三", "李四"])
    rows, _ = stats.summarize(tmp_path, names=["张三"])
    assert [r["姓名"] for r in rows] == ["张三"]


def test_summarize_empty_directory(tmp_path):
    rows, keys = stats.summarize(tmp_path)
    assert rows == [] and keys == []


def test_write_stats_csv(tmp_path):
    rows = [{"姓名": "张三", "应到": 3, "实到": 2, "迟到": 1, "请假": 0,
             "缺勤": 1, "出勤率": "66.7%"}]
    path = stats.write_stats_csv(rows, tmp_path / "out" / "统计.csv")
    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert lines[0] == "学生姓名,应到次数,实到次数,迟到次数,请假次数,缺勤次数,出勤率"
    assert lines[1] == "张三,3,2,1,0,1,66.7%"
