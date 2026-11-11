# -*- coding: utf-8 -*-
"""名单侧栏渲染测试（依赖 Pillow，缺失时自动跳过）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from utils import drawing

pytestmark = pytest.mark.skipif(not drawing._PIL_AVAILABLE, reason="未安装 Pillow")


def _face_photo(path):
    """造一张纯色测试照片"""
    from PIL import Image
    Image.new("RGB", (120, 120), (200, 160, 140)).save(path)


def test_panel_size_matches_row_count(tmp_path):
    panel = drawing.RosterPanel(faces_dir=tmp_path)
    img = panel.render(["张三", "李四"], set())
    expected_h = 78 + drawing.RosterPanel.ROW_H * 2 + 12
    assert img.shape == (expected_h, panel.width, 3)


def test_panel_is_cached_between_frames(tmp_path):
    panel = drawing.RosterPanel(faces_dir=tmp_path)
    first = panel.render(["张三"], set())
    second = panel.render(["张三"], set())
    assert first is second          # 状态未变时不重复绘制，保证帧率


def test_panel_redraws_when_checkin_status_changes(tmp_path):
    panel = drawing.RosterPanel(faces_dir=tmp_path)
    before = panel.render(["张三"], set()).copy()
    after = panel.render(["张三"], {"张三"})
    assert not np.array_equal(before, after)   # 打勾后画面必须变化


def test_thumbnail_loaded_for_registered_student(tmp_path):
    _face_photo(tmp_path / "张三.jpg")
    panel = drawing.RosterPanel(faces_dir=tmp_path)
    assert panel._thumb("张三") is not None
    assert panel._thumb("李四") is None        # 没有照片时用占位圆圈


def test_panel_handles_many_names_with_overflow(tmp_path):
    names = [f"学生{i}" for i in range(20)]
    panel = drawing.RosterPanel(faces_dir=tmp_path)
    img = panel.render(names, set())
    # 只显示前 MAX_ROWS 行，并额外一行提示
    assert img.shape[0] >= 78 + drawing.RosterPanel.ROW_H * drawing.RosterPanel.MAX_ROWS


def test_attach_panel_extends_canvas(tmp_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    panel_img = drawing.RosterPanel(faces_dir=tmp_path).render(["张三"], set())
    out = drawing.attach_roster_panel(frame, panel_img)
    assert out.shape[1] == 640 + panel_img.shape[1]
    assert out.shape[0] >= 480


def test_attach_panel_tolerates_missing_panel():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert drawing.attach_roster_panel(frame, None) is frame
