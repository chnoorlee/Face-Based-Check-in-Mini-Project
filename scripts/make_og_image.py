# -*- coding: utf-8 -*-
"""
宣传页社交分享图（og:image）生成工具

生成 docs/assets/og.png（1200x630），纯 PIL 绘制，无外部依赖：
深色渐变背景 + 检测框元素 + 项目标题。

用法：python scripts/make_og_image.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H = 1200, 630
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "og.png"

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑粗体
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def load_font(size):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def main():
    img = Image.new("RGB", (W, H), (10, 14, 26))
    draw = ImageDraw.Draw(img)

    # 对角渐变背景
    top, bottom = (10, 14, 26), (24, 20, 52)
    for y in range(H):
        draw.line([(0, y), (W, y)], fill=lerp(top, bottom, y / H))

    # 网格纹理
    for x in range(0, W, 48):
        draw.line([(x, 0), (x, H)], fill=(148, 163, 216, 14), width=1)
    for y in range(0, H, 48):
        draw.line([(0, y), (W, y)], fill=(148, 163, 216, 14), width=1)

    # 右上光斑
    for r in range(260, 0, -4):
        alpha = int(26 * (1 - r / 260))
        draw.ellipse([W - 480 - r, -320 + (260 - r), W - 480 + r, -320 + (260 - r) + 2 * r],
                     fill=(124, 108, 255, alpha))

    # 中央人脸检测框（四角高亮）
    bx0, by0, bx1, by1 = 560, 150, 980, 480
    corner = 46
    width = 5
    accent = (124, 108, 255)
    for (cx, cy, dx, dy) in [
        (bx0, by0, 1, 1), (bx1, by0, -1, 1), (bx0, by1, 1, -1), (bx1, by1, -1, -1),
    ]:
        draw.line([(cx + dx * corner, cy), (cx, cy), (cx, cy + dy * corner)], fill=accent, width=width)
        draw.line([(cx + dx * 22, cy), (cx, cy), (cx, cy + dy * 22)], fill=(61, 215, 242), width=width)
    # 虚线连接框边
    for x in range(bx0, bx1, 16):
        draw.line([(x, by0), (x + 8, by0)], fill=(148, 163, 216), width=2)
        draw.line([(x, by1), (x + 8, by1)], fill=(148, 163, 216), width=2)
    for y in range(by0, by1, 16):
        draw.line([(bx0, y), (bx0, y + 8)], fill=(148, 163, 216), width=2)
        draw.line([(bx1, y), (bx1, y + 8)], fill=(148, 163, 216), width=2)

    # 框内抽象人脸（圆头 + 肩部弧线）
    cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2 - 6
    draw.ellipse([cx - 92, cy - 92, cx + 92, cy + 92], outline=(232, 236, 248), width=5)
    for eye_dx in (-36, 36):
        draw.ellipse([cx + eye_dx - 9, cy - 20 - 9, cx + eye_dx + 9, cy - 20 + 9],
                     fill=(61, 215, 242))
    draw.arc([cx - 34, cy + 8, cx + 34, cy + 56], start=20, end=160, fill=(232, 236, 248), width=5)
    draw.arc([cx - 130, by1 - 18, cx + 130, by1 + 190], start=180, end=360,
             fill=(232, 236, 248), width=5)

    # 标签「张三 · 签到成功」
    label = "张三 · 签到成功 ✓"
    f_label = load_font(30)
    tw = draw.textlength(label, font=f_label)
    pad = 14
    lx0, ly0 = bx0 + 2, by0 - 58
    draw.rounded_rectangle([lx0, ly0, lx0 + tw + pad * 2, ly0 + 52], radius=10,
                           fill=(26, 34, 58), outline=(61, 214, 140), width=2)
    draw.text((lx0 + pad, ly0 + 8), label, font=f_label, fill=(61, 214, 140))

    # 左侧标题区
    f_title = load_font(64)
    f_sub = load_font(32)
    draw.text((84, 208), "课堂自动人脸", font=f_title, fill=(232, 236, 248))
    # 第二行标题：渐变文字（用蒙版把渐变层贴进字形）
    title2 = "签到系统"
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).text((84, 292), title2, font=f_title, fill=255)
    grad_layer = Image.new("RGB", (W, H))
    gd = ImageDraw.Draw(grad_layer)
    for i in range(W):
        gd.line([(i, 0), (i, H)], fill=lerp((124, 108, 255), (61, 215, 242), i / (W - 1)))
    img.paste(grad_layer, (0, 0), mask)
    draw.text((86, 400), "YOLO-Face 人脸检测 · SFace 特征比对", font=f_sub, fill=(151, 161, 189))
    draw.text((86, 452), "照片注册 → 摄像头自动签到 · 防重复 · CSV 记录", font=f_sub, fill=(151, 161, 189))

    # 左下角三个模型 chip
    f_chip = load_font(24)
    chips = ["YOLOv8-Face 6MB", "SFace 37MB", "YuNet 230KB"]
    chip_x = 84
    for chip in chips:
        cw = draw.textlength(chip, font=f_chip)
        draw.rounded_rectangle([chip_x, 528, chip_x + cw + 36, 576], radius=24,
                               fill=(18, 26, 48), outline=(124, 108, 255), width=2)
        draw.text((chip_x + 18, 538), chip, font=f_chip, fill=(196, 187, 255))
        chip_x += int(cw) + 54

    img.save(OUT)
    print("已生成:", OUT)


if __name__ == "__main__":
    main()
