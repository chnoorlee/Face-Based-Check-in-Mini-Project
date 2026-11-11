# -*- coding: utf-8 -*-
"""
演示视频生成工具

把一张人脸照片生成短视频，用于无摄像头环境下演示/测试签到流程：
    1. python register.py -i 照片.jpg -n 姓名      # 先注册
    2. python scripts/make_demo_video.py 照片.jpg demo.mp4
    3. python checkin.py --source demo.mp4 --no-window

用法：python scripts/make_demo_video.py 照片.jpg 输出.mp4 [帧数]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2  # noqa: E402

from utils.imutils import imread_unicode  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print("用法：python scripts/make_demo_video.py 照片.jpg 输出.mp4 [帧数]")
        return 1
    img = imread_unicode(sys.argv[1])
    if img is None:
        print("无法读取图片:", sys.argv[1])
        return 1
    try:
        frames = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    except ValueError:
        print("帧数必须是整数")
        return 1
    fps = 10.0
    out = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (img.shape[1], img.shape[0]))
    if not out.isOpened():
        print("无法创建视频文件:", sys.argv[2])
        return 1
    for _ in range(frames):
        out.write(img)
    out.release()
    print(f"已生成演示视频：{sys.argv[2]}（{frames} 帧 @ {fps:.0f}fps）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
