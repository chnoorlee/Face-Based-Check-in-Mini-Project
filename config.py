# -*- coding: utf-8 -*-
"""
全局配置文件 —— 课程 3320 课堂自动人脸签到系统

集中管理目录、模型路径、识别阈值、冷却时间等所有可调参数。
修改本文件后重新运行程序即可生效，无需改动其他代码。
"""
from pathlib import Path

# ---------- 目录 ----------
BASE_DIR = Path(__file__).resolve().parent          # 项目根目录
DATA_DIR = BASE_DIR / "data"                        # 所有本地数据
FACES_DIR = DATA_DIR / "faces"                      # 注册的人脸照片
EMBEDDINGS_DIR = DATA_DIR / "embeddings"            # 人脸特征向量库（每人一个 .npy 文件）
ATTENDANCE_DIR = DATA_DIR / "attendance"            # 签到 CSV 记录（每天一个文件）
LOG_DIR = DATA_DIR / "logs"                         # 运行日志
MODELS_DIR = BASE_DIR / "models"                    # 模型文件目录

# ---------- 模型文件 ----------
# 所有模型文件不存在时都会自动下载（见 utils/face_detector.py 与 utils/face_embedder.py），
# 也可以手动下载后放到 models/ 目录。下载地址列表按顺序尝试，失败自动切换备用地址。
YOLO_FACE_MODEL = MODELS_DIR / "yolov8n-face.pt"                                  # 人脸检测（约 6MB）
YOLO_FACE_MODEL_URLS = [
    "https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
    "https://hf-mirror.com/arnabdhar/YOLOv8-Face-detection/resolve/main/model.pt",
    "https://huggingface.co/arnabdhar/YOLOv8-Face-detection/resolve/main/model.pt",
]
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"                  # 特征提取（约 37MB）
# 注意：opencv_zoo 的模型是 Git LFS 存储，raw.githubusercontent.com 只返回指针文件，
# 必须优先使用 media.githubusercontent.com（官方 LFS 内容端点）
SFACE_MODEL_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"                    # 人脸对齐关键点（约 230KB）
YUNET_MODEL_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://cdn.jsdelivr.net/gh/opencv/opencv_zoo@main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

# ---------- 识别参数 ----------
# 人脸特征距离阈值（欧氏距离，特征已 L2 归一化，取值 0~2），越小越严格：
#   经常认不出自己 -> 调大到 1.2；把别人误认成自己 -> 调小到 0.8
MATCH_TOLERANCE = 1.0
# 同一学生签到后的冷却时间（秒），冷却期内不会重复签到
COOLDOWN_SECONDS = 60
# YOLO 检测置信度阈值（低于该值的检测框会被丢弃）
DETECT_CONFIDENCE = 0.5
# YOLO 推理图像尺寸（越大越准、越慢）
DETECT_IMG_SIZE = 640

# ---------- 摄像头与显示 ----------
CAMERA_INDEX = 0         # 摄像头编号：笔记本内置一般为 0，外接摄像头可改为 1
SHOW_WINDOW = True       # 是否显示实时画面窗口（无图形环境时改为 False）
WINDOW_NAME = "课堂人脸签到 - 按 q 退出"


def validate_config():
    """启动前校验配置取值范围，返回警告信息列表（空列表表示全部合法）。"""
    warnings = []
    if not 0 < MATCH_TOLERANCE <= 2:
        warnings.append(f"MATCH_TOLERANCE={MATCH_TOLERANCE} 超出合理范围 (0, 2]（归一化特征欧氏距离取值 0~2）")
    if COOLDOWN_SECONDS <= 0:
        warnings.append(f"COOLDOWN_SECONDS={COOLDOWN_SECONDS} 必须为正数")
    if not 0 < DETECT_CONFIDENCE <= 1:
        warnings.append(f"DETECT_CONFIDENCE={DETECT_CONFIDENCE} 必须在 (0, 1] 范围内（置信度概率）")
    if not isinstance(CAMERA_INDEX, int) or CAMERA_INDEX < -1:
        warnings.append(f"CAMERA_INDEX={CAMERA_INDEX!r} 应为非负整数（外接摄像头一般用 0 或 1）")
    return warnings
