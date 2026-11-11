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
from utils.imutils import ensure_dirs, imread_unicode  # noqa: E402
from utils.storage import FeatureDatabase  # noqa: E402

logger = logging.getLogger(__name__)


def register_face(image_path: str, name: str):
    """人脸注册核心流程。返回 (是否成功, 提示信息)。

    注册阶段输入是静态图片；签到阶段才是摄像头实时画面。
    """
    name = (name or "").strip()
    if not name:
        return False, "学生姓名不能为空"

    img = imread_unicode(image_path)   # 用 imread_unicode 支持中文路径
    if img is None:
        return False, f"无法读取图片：{image_path}（请检查路径是否正确，中文路径也支持）"

    detector = FaceDetector()
    boxes = detector.detect(img)
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

    db = FeatureDatabase()
    overwritten = db.exists(name)
    db.save(name, embedding, img[y1:y2, x1:x2])

    detail = f"注册成功！姓名：{name} | 检测后端：{detector.backend}"
    if overwritten:
        detail += "（已覆盖同名旧记录）"
    return True, detail


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="课堂人脸签到系统 - 人脸注册模块")
    parser.add_argument("--image", "-i", help="人脸照片路径（支持中文路径）")
    parser.add_argument("--name", "-n", help="学生姓名")
    args = parser.parse_args()

    # 未通过命令行传参时，交互式询问
    image_path = args.image or input("请输入人脸照片路径（可把文件拖进窗口）: ").strip().strip('"')
    name = args.name or input("请输入学生姓名: ").strip()

    ok, msg = register_face(image_path, name)
    print(("✅ " if ok else "❌ ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
