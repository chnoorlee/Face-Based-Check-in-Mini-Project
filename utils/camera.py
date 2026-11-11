# -*- coding: utf-8 -*-
"""
摄像头健康检查工具

真实调试中发现的问题：笔记本摄像头的自动曝光需要 1~2 秒才稳定，
只预热几帧时画面接近全黑；而隐私挡板关闭时画面会是一片纯色（无任何结构），
此时签到程序只会「一直检测不到人脸」，用户完全不知道原因。

本模块负责：
  - warmup()：按时间预热到曝光稳定
  - analyze_frame() / diagnose_frame()：判断画面是否可正常识别，并给出可操作的提示
"""
import time

import cv2
import numpy as np

# 判定阈值
_EDGE_RATIO_MIN = 0.003      # 边缘像素占比低于此值 → 画面无结构（纯色/被遮挡）
_BRIGHTNESS_MIN = 30         # 亮度过低 → 环境太暗或挡板关闭
_BRIGHTNESS_MAX = 235        # 亮度过高 → 严重过曝
_COLORFULNESS_MIN = 1.5      # 彩色度极低且无边缘 → 纯灰画面


def warmup(cap, min_seconds=1.5, max_seconds=3.0, stable_frames=5, tolerance=2.0):
    """预热摄像头直到自动曝光稳定。

    实测：首帧亮度可能为 0~4，约 1~2 秒后才升到正常值；固定读 5 帧几乎等于没预热，
    而固定等 1.5 秒又可能刚好卡在曝光跳变的瞬间。因此：
      至少等 min_seconds，之后连续 stable_frames 帧亮度变化小于 tolerance 即认为稳定，
      最多等 max_seconds。

    返回 (读取帧数, 最后一帧)；读不到画面时最后一帧为 None。
    """
    start = time.monotonic()
    frames, prev, stable, last = 0, None, 0, None
    while time.monotonic() - start < max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        last = frame
        brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        elapsed = time.monotonic() - start
        # 全黑画面同样「稳定」，但那只是曝光还没跟上，不能据此提前结束预热
        usable = brightness >= _BRIGHTNESS_MIN
        if prev is not None and abs(brightness - prev) <= tolerance and usable:
            stable += 1
            if elapsed >= min_seconds and stable >= stable_frames:
                break
        else:
            stable = 0
        prev = brightness
    return frames, last


def analyze_frame(frame):
    """返回画面质量指标：亮度、边缘占比、彩色度"""
    if frame is None or frame.size == 0:
        return {"brightness": 0.0, "edge_ratio": 0.0, "colorfulness": 0.0}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    b, g, r = cv2.split(frame)
    # 彩色度：色度通道相对亮度的偏离程度（纯灰画面接近 0）
    b, g, r = b.astype(np.int16), g.astype(np.int16), r.astype(np.int16)
    colorfulness = float((np.abs(b - g).mean() + np.abs(r - g).mean()) / 2.0)
    return {
        "brightness": float(gray.mean()),
        "edge_ratio": float((edges > 0).mean()),
        "colorfulness": colorfulness,
    }


def diagnose_frame(frame):
    """判断画面是否适合做人脸识别。

    返回 (是否正常, 提示文字)。不正常时给出具体、可操作的原因。
    """
    stats = analyze_frame(frame)
    brightness = stats["brightness"]
    edge_ratio = stats["edge_ratio"]
    colorful = stats["colorfulness"]

    # 先判断明暗（最直观、用户最容易自己解决），再判断「有亮度但没有任何结构」
    if brightness < _BRIGHTNESS_MIN:
        return False, f"画面过暗（亮度 {brightness:.0f}），请开灯或调整摄像头朝向"
    if brightness > _BRIGHTNESS_MAX:
        return False, f"画面过曝（亮度 {brightness:.0f}），请避免强光直射镜头"
    if edge_ratio < _EDGE_RATIO_MIN and colorful < _COLORFULNESS_MIN:
        return False, (
            f"摄像头画面几乎是纯色（亮度 {brightness:.0f}，边缘占比 {edge_ratio * 100:.2f}%）。"
            "请检查：① 笔记本摄像头的物理隐私挡板是否关闭；"
            "② 镜头是否被贴纸/灰尘遮挡；③ 是否选错了摄像头（config.py 的 CAMERA_INDEX）"
        )
    return True, f"画面正常（亮度 {brightness:.0f}，边缘占比 {edge_ratio * 100:.1f}%）"
