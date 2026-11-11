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
            dist = f" {det.distance:.2f}" if det.distance is not None else ""
            text = f"{det.name} · {_STATUS_TEXT[det.status]}{dist}"
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
