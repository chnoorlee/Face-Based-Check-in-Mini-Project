# -*- coding: utf-8 -*-
"""
人脸特征提取模块

使用 OpenCV DNN 模块（opencv-python 自带，无需安装额外深度学习框架）运行两个轻量模型：
  1. YuNet（face_detection_yunet_2023mar.onnx，约 230KB）：
     在 YOLO-Face 检测到的人脸区域上提取 5 个关键点，用于人脸对齐；
  2. SFace（face_recognition_sface_2021dec.onnx，约 37MB）：
     把对齐后的人脸编码为 128 维特征向量（L2 归一化）。

两张人脸是否同一人用特征向量的欧氏距离衡量：距离越小越可能是同一个人。
选择 OpenCV DNN 方案的原因：全部依赖 pip 即可安装、无需编译 dlib/CMake，
对课程作业的"开箱即用"最友好。
"""
import logging

import cv2
import numpy as np

from config import MODELS_DIR, SFACE_MODEL, SFACE_MODEL_URLS, YUNET_MODEL, YUNET_MODEL_URLS
from utils.imutils import download_file

logger = logging.getLogger(__name__)

# SFace 模型要求的输入尺寸
_FACE_SIZE = 112
# 人脸区域小于该尺寸（像素）时认为质量不足，跳过特征提取
_MIN_FACE_SIZE = 24
# YuNet 关键点检测置信度
_YUNET_SCORE = 0.6

# 对齐目标模板：112x112 下标准五点的平均位置（左眼/右眼/鼻尖/左嘴角/右嘴角）
_ALIGN_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


class FaceEmbedder:
    """人脸特征提取器（惰性加载模型：首次使用时自动下载并加载）"""

    def __init__(self):
        self._sface = None
        self._yunet = None

    def prepare(self):
        """下载（如需）并加载模型。失败抛 RuntimeError（信息可直接展示给用户）。"""
        if self._sface is not None:
            return self
        # min_size 用于识别下载到 Git LFS 指针文件（约 130 字节的文本）等无效内容
        if not download_file(SFACE_MODEL, SFACE_MODEL_URLS, "SFace（约 37MB）", min_size=1024 * 1024):
            raise RuntimeError("SFace 特征模型下载失败，请手动下载放到 models/ 目录（详见 README）")
        if not download_file(YUNET_MODEL, YUNET_MODEL_URLS, "YuNet（约 230KB）", min_size=64 * 1024):
            raise RuntimeError("YuNet 关键点模型下载失败，请手动下载放到 models/ 目录（详见 README）")
        try:
            self._sface = cv2.dnn.readNetFromONNX(str(SFACE_MODEL))
            self._yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320),
                                                    _YUNET_SCORE, 0.3, 5000)
            logger.info("SFace / YuNet 模型加载成功")
        except cv2.error as exc:
            raise RuntimeError(f"人脸特征模型加载失败：{exc}") from exc
        return self

    def extract(self, frame_bgr, box):
        """提取 box=(x1,y1,x2,y2) 区域内人脸的 128 维归一化特征向量。

        成功返回 numpy 数组；人脸过小或提取失败返回 None。
        """
        if self._sface is None:
            self.prepare()
        aligned = self._align_face(frame_bgr, box)
        if aligned is None:
            return None
        # 注意：SFace 的 ONNX 导出期望 0~255 的原始像素输入，这里不能除以 255！
        # （实测除以 255 后输入近似归零，不同人之间的距离会退化到 0.8 左右，无法区分）
        blob = cv2.dnn.blobFromImage(aligned, 1.0, (_FACE_SIZE, _FACE_SIZE),
                                     (0, 0, 0), swapRB=True)
        self._sface.setInput(blob)
        vec = self._sface.forward().flatten()
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return (vec / norm).astype(np.float32)

    def _align_face(self, frame_bgr, box):
        """裁剪人脸区域并用 YuNet 关键点做相似变换对齐；对齐失败则直接缩放。"""
        x1, y1, x2, y2 = (int(v) for v in box)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < _MIN_FACE_SIZE or y2 - y1 < _MIN_FACE_SIZE:
            return None
        crop = frame_bgr[y1:y2, x1:x2]

        # 尝试关键点对齐（YuNet 输入尺寸需与图像匹配）
        self._yunet.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self._yunet.detect(crop)
        if faces is not None and len(faces) > 0:
            landmarks = faces[0][4:14].reshape(5, 2).astype(np.float32)
            transform = cv2.estimateAffinePartial2D(landmarks, _ALIGN_TEMPLATE)[0]
            if transform is not None:
                return cv2.warpAffine(crop, transform, (_FACE_SIZE, _FACE_SIZE))
        # 对齐失败：直接缩放（对正面人脸影响不大）
        return cv2.resize(crop, (_FACE_SIZE, _FACE_SIZE))

    @staticmethod
    def distance(vec_a, vec_b) -> float:
        """两个特征向量的欧氏距离"""
        return float(np.linalg.norm(np.asarray(vec_a) - np.asarray(vec_b)))
