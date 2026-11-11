# -*- coding: utf-8 -*-
"""
画面标注工具
- 在人脸周围绘制识别框与标签（姓名 / 未知人员 / 签到状态）
- 使用 PIL 绘制文字以支持中文姓名；PIL 或中文字体不可用时回退 OpenCV 英文绘制
"""
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# 状态 -> 画面标签文字
_STATUS_TEXT = {
    "new": "签到成功",
    "cooldown": "冷却中",
    "checked_in": "已签到",
    "unknown": "未知人员",
}

# 常见中文字体路径（按平台尝试）
_FONT_PATH_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",                              # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",                            # Windows 黑体
    "/System/Library/Fonts/PingFang.ttc",                     # macOS 苹方
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # Linux Noto
]

_FONT_CACHE = {}


def _load_font(size: int = 22):
    """加载中文字体（带缓存），失败返回 None"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    if _PIL_AVAILABLE:
        for path in _FONT_PATH_CANDIDATES:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
    _FONT_CACHE[size] = font
    return font


def _put_text_fallback(frame, text, pos, color_bgr):
    """OpenCV 绘制回退方案（不支持中文，中文会显示为问号）"""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)


class RosterPanel:
    """签到名单侧栏：显示应到学生与打勾状态。

    画面右侧显示「应到 X 人 / 已到 Y 人」和逐人勾选状态，签到成功即打勾。
    侧栏图像带缓存，只有签到状态变化时才重绘，不影响实时帧率。
    """

    ROW_H = 34
    MAX_ROWS = 18   # 名单过长时只显示前 N 人，其余在控制台查看

    def __init__(self, title="签到名单", width=250):
        self.title = title
        self.width = width
        self._cache_key = None
        self._cache_img = None

    def render(self, names, checked):
        """返回侧栏图像（OpenCV BGR 格式）"""
        key = (tuple(names), tuple(n for n in names if n in checked))
        if key != self._cache_key:
            self._cache_img = self._build(names, checked)
            self._cache_key = key
        return self._cache_img

    def _build(self, names, checked):
        shown = list(names[: self.MAX_ROWS])
        overflow = len(names) - len(shown)
        height = 78 + self.ROW_H * len(shown) + (28 if overflow > 0 else 0) + 12

        img = Image.new("RGB", (self.width, height), (18, 24, 38))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, self.width - 1, height - 1], outline=(48, 60, 92))

        present = sum(1 for n in names if n in checked)
        draw.text((14, 12), self.title, font=_load_font(20) or ImageFont.load_default(),
                  fill=(232, 236, 248))
        draw.text((14, 42), f"已到 {present} / {len(names)} 人",
                  font=_load_font(16) or ImageFont.load_default(), fill=(151, 161, 189))

        font = _load_font(17) or ImageFont.load_default()
        y = 78
        for name in shown:
            done = name in checked
            color = (61, 214, 140) if done else (110, 120, 145)
            # 勾选框
            draw.rounded_rectangle([14, y + 5, 31, y + 22], radius=5, outline=color, width=2)
            if done:   # 打勾
                draw.line([(17, y + 14), (21, y + 18), (28, y + 9)], fill=color, width=3)
            draw.text((41, y), name, font=font,
                      fill=(232, 236, 248) if done else (151, 161, 189))
            y += self.ROW_H

        if overflow > 0:
            draw.text((14, y + 2), f"…另有 {overflow} 人（见控制台）",
                      font=_load_font(14) or ImageFont.load_default(), fill=(110, 120, 145))

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def attach_roster_panel(frame, panel_bgr):
    """把名单侧栏拼接到画面右侧"""
    if panel_bgr is None:
        return frame
    h = max(frame.shape[0], panel_bgr.shape[0])
    canvas = np.full((h, frame.shape[1] + panel_bgr.shape[1], 3), (18, 24, 38), dtype=np.uint8)
    canvas[: frame.shape[0], : frame.shape[1]] = frame
    canvas[: panel_bgr.shape[0], frame.shape[1]:] = panel_bgr
    return canvas


def draw_detections(frame, detections):
    """在画面上绘制所有识别结果（人脸框 + 标签）。

    detections 元素需具备属性：box / name / distance / status
    （定义见 checkin.Detection）
    """
    labels = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        if det.name is None:
            color = (0, 0, 255)          # 红色：未知人员
            text = _STATUS_TEXT["unknown"]
        else:
            color = (0, 200, 0) if det.status == "new" else (0, 180, 255)  # 绿=新签到 黄=已签到/冷却
            if det.status == "cooldown" and getattr(det, "cooldown_left", None) is not None:
                status_txt = f"{_STATUS_TEXT['cooldown']} {max(int(det.cooldown_left), 0)}s"
            else:
                status_txt = _STATUS_TEXT[det.status]
            dist = f" {det.distance:.2f}" if det.distance is not None else ""
            text = f"{det.name} · {status_txt}{dist}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        labels.append((text, (x1, max(y1 - 28, 2)), color))

    if not labels:
        return frame

    font = _load_font()
    if font is None:
        for text, pos, color in labels:
            _put_text_fallback(frame, text, pos, color)
        return frame

    # 一次性转为 PIL 绘制全部中文标签，再转回 OpenCV 格式
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, pos, color in labels:
        draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return frame
