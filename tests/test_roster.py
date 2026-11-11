# -*- coding: utf-8 -*-
"""utils.roster.Roster 应到名单与出勤统计单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from utils.roster import Roster


class TestConstruction:
    def test_from_database_uses_all_registered(self):
        roster = Roster.from_database({"张三": None, "李四": None})
        assert roster.names == ["张三", "李四"]
        assert len(roster) == 2

    def test_strips_and_dedupes(self):
        roster = Roster(["  张三 ", "张三", "", "李四", None])
        assert roster.names == ["张三", "李四"]

    def test_txt_file(self, tmp_path):
        p = tmp_path / "名单.txt"
        p.write_text("张三\n李四\n# 注释行\n\n王五\n", encoding="utf-8")
        assert Roster.from_file(p).names == ["张三", "李四", "王五"]

    def test_csv_file_with_header(self, tmp_path):
        p = tmp_path / "名单.csv"
        p.write_text("学生姓名,学号\n张三,001\n李四,002\n", encoding="utf-8-sig")
        assert Roster.from_file(p).names == ["张三", "李四"]

    def test_csv_quoted_field_with_comma(self, tmp_path):
        # 姓名含逗号时 Excel 会加引号，必须用标准 csv 解析
        p = tmp_path / "名单.csv"
        p.write_text('学生姓名,备注\n"张,三",\n李四,\n', encoding="utf-8-sig")
        assert Roster.from_file(p).names == ["张,三", "李四"]

    def test_txt_keeps_full_line(self, tmp_path):
        p = tmp_path / "名单.txt"
        p.write_text("张三\n李四\n", encoding="utf-8")
        assert Roster.from_file(p).names == ["张三", "李四"]

    def test_gbk_encoded_file(self, tmp_path):
        p = tmp_path / "名单.txt"
        p.write_bytes("张三\n李四\n".encode("gbk"))
        assert Roster.from_file(p).names == ["张三", "李四"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Roster.from_file(tmp_path / "不存在.txt")

    def test_csv_leave_marker(self, tmp_path):
        p = tmp_path / "名单.csv"
        p.write_text("学生姓名,状态\n张三,请假\n李四,\n", encoding="utf-8-sig")
        roster = Roster.from_file(p)
        assert roster.names == ["张三", "李四"]
        assert roster.on_leave() == ["张三"]
        assert roster.absent(set()) == ["李四"]

    def test_explicit_leave_argument(self):
        roster = Roster(["张三", "李四", "王五"], leave=["李四"])
        assert roster.on_leave() == ["李四"]
        assert roster.absent(set()) == ["张三", "王五"]

    def test_leave_names_outside_roster_ignored(self):
        roster = Roster(["张三"], leave=["赵六"])
        assert roster.on_leave() == []


class TestAttendanceStats:
    def setup_method(self):
        self.roster = Roster(["张三", "李四", "王五"])

    def test_present_and_absent(self):
        checked = {"张三", "王五"}
        assert self.roster.present(checked) == ["张三", "王五"]
        assert self.roster.absent(checked) == ["李四"]

    def test_nobody_checked_in(self):
        assert self.roster.present(set()) == []
        assert self.roster.absent(set()) == ["张三", "李四", "王五"]

    def test_everybody_checked_in(self):
        all_checked = {"张三", "李四", "王五"}
        assert self.roster.absent(all_checked) == []

    def test_checked_names_outside_roster_ignored(self):
        # 特征库里有、但不在本次名单上的学生，不影响出勤统计
        assert self.roster.present({"张三", "赵六"}) == ["张三"]
        assert self.roster.absent({"张三", "赵六"}) == ["李四", "王五"]

    def test_progress_text(self):
        assert self.roster.progress_text({"张三"}) == "1/3"

    def test_unregistered_names(self):
        db = {"张三": None}
        assert self.roster.unregistered(db) == ["李四", "王五"]
