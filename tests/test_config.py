# -*- coding: utf-8 -*-
"""config.validate_config 配置校验单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def test_default_config_is_valid():
    assert config.validate_config() == []


def test_invalid_tolerance_warns(monkeypatch):
    monkeypatch.setattr(config, "MATCH_TOLERANCE", 3.0)
    warnings = config.validate_config()
    assert any("MATCH_TOLERANCE" in w for w in warnings)


def test_nonpositive_cooldown_warns(monkeypatch):
    monkeypatch.setattr(config, "COOLDOWN_SECONDS", 0)
    warnings = config.validate_config()
    assert any("COOLDOWN_SECONDS" in w for w in warnings)


def test_invalid_confidence_warns(monkeypatch):
    monkeypatch.setattr(config, "DETECT_CONFIDENCE", 1.5)
    warnings = config.validate_config()
    assert any("DETECT_CONFIDENCE" in w for w in warnings)


def test_invalid_camera_index_warns(monkeypatch):
    monkeypatch.setattr(config, "CAMERA_INDEX", -5)
    warnings = config.validate_config()
    assert any("CAMERA_INDEX" in w for w in warnings)
