# -*- coding: utf-8 -*-
"""
人脸检测模块

优先使用 YOLO-Face 模型（yolov8n-face.pt）进行人脸检测；
模型文件缺失、自动下载失败或 ultralytics 未安装时，
自动回退到 OpenCV 自带的 Haar 级联检测器，保证程序任何环境下都能运行。
"""
import logging
from pathlib import Path

import cv2

from config import (DETECT_CONFIDENCE, DETECT_IMG_SIZE, YOLO_FACE_MODEL,
                    YOLO_FACE_MODEL_URLS)
from utils.imutils import download_file

logger = logging.getLogger(__name__)


def ensure_model_file(model_path: Path) -> bool:
    """检查 YOLO-Face 模型文件是否完整存在，缺失或过小（下载中断残留）时尝试自动下载。"""
    model_path = Path(model_path)
    if model_path.exists() and model_path.stat().st_size >= 1024 * 1024:
        return True
    if model_path.exists():
        model_path.unlink()   # 清除残留的不完整文件，否则加载会一直失败
    print(f"[模型下载] 未找到 {model_path.name}，正在自动下载（约 6MB）...")
    ok = download_file(model_path, YOLO_FACE_MODEL_URLS, "YOLO-Face（约 6MB）", min_size=1024 * 1024)
    if not ok:
        print("          也可暂时使用 OpenCV Haar 检测继续运行（精度较低）。")
    return ok


class FaceDetector:
    """统一的人脸检测接口。

    用法：
        detector = FaceDetector()
        boxes = detector.detect(frame_bgr)   # [(x1, y1, x2, y2), ...]
    """

    def __init__(self, model_path=None, conf=DETECT_CONFIDENCE, imgsz=DETECT_IMG_SIZE):
        self.conf = conf
        self.imgsz = imgsz
        self.backend = "none"
        self._model = None

        path = Path(model_path) if model_path else YOLO_FACE_MODEL
        if ensure_model_file(path):
            try:
                from ultralytics import YOLO  # 延迟导入，避免拖慢启动
                self._model = YOLO(str(path))
                self.backend = "YOLO-Face"
                logger.info("YOLO-Face 模型加载成功：%s", path)
            except Exception as exc:
                logger.warning("YOLO-Face 模型加载失败（%s），回退到 OpenCV Haar", exc)

        if self._model is None:
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                raise RuntimeError("OpenCV Haar 级联文件缺失，无法初始化人脸检测器")
            self.backend = "OpenCV-Haar"
            logger.warning("未使用 YOLO-Face 模型，当前为 OpenCV Haar 检测（精度较低）")

    def detect(self, frame):
        """检测画面中所有人脸，返回边界框列表 [(x1, y1, x2, y2), ...]"""
        if self.backend == "YOLO-Face":
            return self._detect_yolo(frame)
        return self._detect_haar(frame)

    def _detect_yolo(self, frame):
        boxes = []
        results = self._model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for row in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = row[:4].astype(int)
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return [(x, y, x + w, y + h) for (x, y, w, h) in faces]
