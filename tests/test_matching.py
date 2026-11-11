# -*- coding: utf-8 -*-
"""find_best_match 特征比对阈值与匹配逻辑单元测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.storage import find_best_match


def test_empty_database_returns_none():
    name, dist = find_best_match(np.zeros(4), {}, 0.5)
    assert name is None and dist is None


def test_exact_match():
    db = {"张三": np.array([0.0, 0.0]), "李四": np.array([10.0, 0.0])}
    name, dist = find_best_match(np.array([0.0, 0.0]), db, 0.5)
    assert name == "张三" and dist == 0.0


def test_distance_within_tolerance_matches():
    db = {"张三": np.array([0.0, 0.0])}
    name, dist = find_best_match(np.array([0.4, 0.0]), db, 0.5)
    assert name == "张三" and abs(dist - 0.4) < 1e-9


def test_distance_beyond_tolerance_is_unknown():
    db = {"张三": np.array([0.0, 0.0])}
    name, dist = find_best_match(np.array([0.6, 0.0]), db, 0.5)
    assert name is None and abs(dist - 0.6) < 1e-9


def test_returns_closest_even_when_unknown():
    # 查询向量 [1.1,0] 与李四 [2,0] 距离 0.9、与张三 [0,0] 距离 1.1，
    # 最近的是李四，但 0.9 > 0.5 阈值，仍判定为未知人员
    db = {"张三": np.array([0.0, 0.0]), "李四": np.array([2.0, 0.0])}
    name, dist = find_best_match(np.array([1.1, 0.0]), db, 0.5)
    assert name is None and abs(dist - 0.9) < 1e-9
