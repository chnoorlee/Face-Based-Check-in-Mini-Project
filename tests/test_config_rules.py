# -*- coding: utf-8 -*-
"""考勤规则（上课时间 / 迟到宽限）解析测试"""
import sys
from datetime import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config


def test_parse_class_start():
    assert config.parse_class_start("08:00") == time(8, 0)
    assert config.parse_class_start("13:45") == time(13, 45)


def test_parse_class_start_invalid():
    with pytest.raises(ValueError):
        config.parse_class_start("8点")
    with pytest.raises(ValueError):
        config.parse_class_start("25:00")


def test_late_after_adds_grace(monkeypatch):
    monkeypatch.setattr(config, "CLASS_START_TIME", "08:00")
    monkeypatch.setattr(config, "LATE_GRACE_MINUTES", 10)
    assert config.late_after() == time(8, 10)


def test_late_after_handles_midnight_rollover(monkeypatch):
    monkeypatch.setattr(config, "CLASS_START_TIME", "23:55")
    monkeypatch.setattr(config, "LATE_GRACE_MINUTES", 10)
    assert config.late_after() == time(0, 5)


def test_late_after_returns_none_on_bad_format(monkeypatch):
    assert config.late_after("不是时间") is None


def test_validate_config_checks_start_time(monkeypatch):
    monkeypatch.setattr(config, "CLASS_START_TIME", "8点")
    assert any("CLASS_START_TIME" in w for w in config.validate_config())


def test_validate_config_checks_negative_grace(monkeypatch):
    monkeypatch.setattr(config, "LATE_GRACE_MINUTES", -1)
    assert any("LATE_GRACE_MINUTES" in w for w in config.validate_config())
