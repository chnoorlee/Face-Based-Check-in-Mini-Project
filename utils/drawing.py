# -*- coding: utf-8 -*-
"""
画面标注工具
- 在人脸周围绘制识别框与标签（姓名 / 未知人员 / 签到状态）
- 使用 PIL 绘制文字以支持中文姓名；PIL 或中文字体不可用时回退 OpenCV 英文绘制
"""
from pathlib import Path

import cv2
import numpy as np

from utils.imutils import sanitize_filename

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
    "error": "写入失败",
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
    """签到名单侧栏：显示应到学生、人脸头像与打勾状态。

    画面右侧显示「已到 X / 应到 Y」和逐人条目：
    [头像] 姓名 ............ [✓]
    签到成功即打勾（绿色勾），未签到为灰色空心框。
    侧栏图像带缓存，只有签到状态变化时才重绘，不影响实时帧率。
    """

    ROW_H = 42
    MAX_ROWS = 14   # 名单过长时只显示前 N 人，其余在控制台查看
    THUMB = 34      # 头像直径

    def __init__(self, title="签到名单", width=300, faces_dir=None):
        self.title = title
        self.width = width
        self.faces_dir = Path(faces_dir) if faces_dir else None
        self._cache_key = None
        self._cache_img = None
        self._thumbs = {}
        self._mask = None

    # ---------- 头像 ----------
    def _thumb(self, name):
        """加载学生头像（圆形），带缓存；无照片时返回 None"""
        if name in self._thumbs:
            return self._thumbs[name]
        img = None
        if self.faces_dir is not None:
            # 照片按序号保存：姓名.jpg、姓名_2.jpg……（从 2 开始编号）
            for candidate in (f"{sanitize_filename(name)}.jpg",
                              f"{sanitize_filename(name)}_2.jpg"):
                path = self.faces_dir / candidate
                if path.exists():
                    try:
                        src = Image.open(path).convert("RGB")
                        w, h = src.size
                        side = min(w, h)
                        src = src.crop(((w - side) // 2, (h - side) // 2,
                                        (w + side) // 2, (h + side) // 2))
                        img = src.resize((self.THUMB, self.THUMB), Image.LANCZOS)
                    except (OSError, ValueError):
                        img = None
                    break
        self._thumbs[name] = img
        return img

    def _circle_mask(self):
        if self._mask is None:
            mask = Image.new("L", (self.THUMB, self.THUMB), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, self.THUMB - 1, self.THUMB - 1), fill=255)
            self._mask = mask
        return self._mask

    def render(self, names, checked):
        """返回侧栏图像（OpenCV BGR 格式）；Pillow 不可用时返回 None（不显示侧栏）"""
        if not _PIL_AVAILABLE:
            return None
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
        draw.text((14, 10), self.title, font=_load_font(20) or ImageFont.load_default(),
                  fill=(232, 236, 248))
        draw.text((14, 42), f"已到 {present} / {len(names)} 人",
                  font=_load_font(16) or ImageFont.load_default(), fill=(151, 161, 189))

        font = _load_font(17) or ImageFont.load_default()
        mask = self._circle_mask()
        y = 72
        for name in shown:
            done = name in checked
            color = (61, 214, 140) if done else (110, 120, 145)
            text_color = (232, 236, 248) if done else (151, 161, 189)

            # 已签到：整行淡绿背景，一眼就能看出谁到了
            if done:
                draw.rounded_rectangle([8, y, self.width - 10, y + self.ROW_H - 4],
                                       radius=8, fill=(24, 44, 40))

            # 头像（无照片时画一个灰色圆圈 + 姓名首字）
            thumb = self._thumb(name)
            tx, ty = 14, y + 3
            if thumb is not None:
                img.paste(thumb, (tx, ty), mask)
            else:
                draw.ellipse([tx, ty, tx + self.THUMB - 1, ty + self.THUMB - 1],
                             fill=(40, 48, 70), outline=color, width=1)
                draw.text((tx + 10, ty + 8), name[:1], font=font, fill=text_color)

            # 姓名
            draw.text((tx + self.THUMB + 10, y + 9), name, font=font, fill=text_color)

            # 右侧勾选框
            bx = self.width - 40
            draw.rounded_rectangle([bx, y + 9, bx + 20, y + 29], radius=6,
                                   outline=color, width=2,
                                   fill=(32, 74, 56) if done else None)
            if done:   # 打勾
                draw.line([(bx + 4, y + 19), (bx + 9, y + 24), (bx + 17, y + 13)],
                          fill=color, width=3)
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
            if det.status == "new":
                color = (0, 200, 0)      # 绿：新签到
            elif det.status == "error":
                color = (0, 0, 255)      # 红：写入失败
            else:
                color = (0, 180, 255)    # 黄：已签到 / 冷却中
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
