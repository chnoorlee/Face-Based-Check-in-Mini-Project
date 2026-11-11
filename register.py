# -*- coding: utf-8 -*-
"""
人脸注册模块（静态图片）

流程：读取照片 -> YOLO-Face 人脸检测 -> 提取 128 维人脸特征 -> 保存到本地特征库
特征保存位置：data/embeddings/姓名.npy；人脸照片备份：data/faces/姓名.jpg

用法：
    python register.py --image 照片.jpg --name 张三
    python register.py          # 不带参数时交互式输入
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2

# 保证从任意目录运行都能找到项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文/表情符号输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import LOG_DIR  # noqa: E402
from utils.face_detector import FaceDetector  # noqa: E402
from utils.face_embedder import FaceEmbedder  # noqa: E402
from utils.imutils import ensure_dirs, imread_unicode, sanitize_filename  # noqa: E402
from utils.storage import FeatureDatabase  # noqa: E402

logger = logging.getLogger(__name__)


def register_face(image_path: str, name: str):
    """人脸注册核心流程。返回 (是否成功, 提示信息)。

    注册阶段输入是静态图片；签到阶段才是摄像头实时画面。
    """
    name = (name or "").strip()
    if not name:
        return False, "学生姓名不能为空"
    if len(name) > 50:
        return False, "学生姓名过长（最多 50 个字符）"

    img = imread_unicode(image_path)   # 用 imread_unicode 支持中文路径
    if img is None:
        return False, f"无法读取图片：{image_path}（请检查路径是否正确，中文路径也支持）"

    try:
        detector = FaceDetector()
        boxes = detector.detect(img)
    except (RuntimeError, cv2.error) as exc:   # 检测器初始化/推理失败
        return False, f"人脸检测失败：{exc}"
    if not boxes:
        return False, "未在照片中检测到人脸，请换一张光线充足、正面无遮挡的照片"
    if len(boxes) > 1:
        # 一张照片多张人脸时，默认注册面积最大的一张
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        logger.warning("照片中检测到 %d 张人脸，已自动选择面积最大的一张", len(boxes))

    try:
        embedding = FaceEmbedder().extract(img, boxes[0])
    except RuntimeError as exc:   # 特征模型下载/加载失败
        return False, str(exc)
    if embedding is None:
        return False, "人脸特征提取失败，请使用更清晰的正脸照片重试"

    # 裁剪人脸区域保存照片（边界钳制到画面内，与特征提取的钳制逻辑一致）
    x1, y1, x2, y2 = boxes[0]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False, "人脸区域为空，请换一张照片重试"

    db = FeatureDatabase()
    overwritten = db.exists(name)
    try:
        ok, save_msg = db.save(name, embedding, face_crop)
    except OSError as exc:      # 磁盘/权限问题
        return False, f"保存人脸特征失败：{exc}"
    if not ok:
        return False, save_msg

    detail = f"注册成功！姓名：{name} | 检测后端：{detector.backend}{save_msg}"
    if overwritten:
        detail += "（已覆盖同名旧记录）"
    # 姓名含特殊字符时，文件会被改名保存；提示实际文件名，便于排查
    if sanitize_filename(name) != name:
        detail += (f"\n⚠️ 姓名含特殊字符，特征文件保存为「{sanitize_filename(name)}.npy」；"
                   f"报告与名单中仍按原名显示")
    return True, detail


def main():  # pragma: no cover - 命令行入口（手工实测覆盖）
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from config import validate_config
    for warning in validate_config():
        print("⚠️ 配置警告：", warning)
    parser = argparse.ArgumentParser(description="课堂人脸签到系统 - 人脸注册模块")
    parser.add_argument("--image", "-i", help="人脸照片路径（支持中文路径）")
    parser.add_argument("--name", "-n", help="学生姓名")
    parser.add_argument("--list", "-l", action="store_true", help="列出已注册学生与注册时间")
    parser.add_argument("--delete", "-d", metavar="姓名", help="删除已注册学生")
    args = parser.parse_args()

    db = FeatureDatabase()
    if args.list:
        infos = db.details()
        if not infos:
            print("特征库为空，还没有注册学生")
            return 0
        print(f"已注册学生（{len(infos)} 人）：")
        for info in infos:
            print(f"  - {info['name']}（注册于 {info['registered_at']}）")
        return 0

    if args.delete:
        removed = db.delete(args.delete)
        if removed:
            print(f"✅ 已删除学生：{args.delete}")
            return 0
        print(f"❌ 特征库中没有学生：{args.delete}")
        return 1

    # 未通过命令行传参时，交互式询问
    image_path = args.image or input("请输入人脸照片路径（可把文件拖进窗口）: ").strip().strip('"')
    name = args.name or input("请输入学生姓名: ").strip()

    ok, msg = register_face(image_path, name)
    print(("✅ " if ok else "❌ ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
