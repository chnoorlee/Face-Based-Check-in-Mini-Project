# -*- coding: utf-8 -*-
"""register.register_face 单元测试（用假检测器/假特征器，不加载真实模型）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pytest

import register
import utils.storage as storage_mod


class FakeDetector:
    """假检测器：返回预设的人脸框"""
    def __init__(self, boxes):
        self.boxes = boxes
        self.backend = "fake"

    def detect(self, img):
        return list(self.boxes)


class FakeEmbedder:
    """假特征器：固定返回全 1 向量"""
    def extract(self, img, box):
        return np.ones(128)


@pytest.fixture
def tmp_db_dirs(tmp_path, monkeypatch):
    """把特征库目录重定向到临时目录，避免污染真实数据"""
    monkeypatch.setattr(storage_mod, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(storage_mod, "FACES_DIR", tmp_path / "faces")
    return tmp_path


def make_image(tmp_path, name="face.jpg"):
    img_path = tmp_path / name
    cv2.imwrite(str(img_path), np.zeros((80, 80, 3), dtype=np.uint8))
    return str(img_path)


def test_register_success_writes_files(tmp_db_dirs, monkeypatch):
    monkeypatch.setattr(register, "FaceDetector", lambda: FakeDetector([(10, 10, 60, 60)]))
    monkeypatch.setattr(register, "FaceEmbedder", lambda: FakeEmbedder())
    img = make_image(tmp_db_dirs)

    ok, msg = register.register_face(img, "张三")

    assert ok and "注册成功" in msg
    assert (tmp_db_dirs / "emb" / "张三.npy").exists()
    assert (tmp_db_dirs / "faces" / "张三.jpg").exists()


def test_register_empty_name(tmp_db_dirs):
    ok, msg = register.register_face("whatever.jpg", "")
    assert not ok and "不能为空" in msg


def test_register_name_too_long(tmp_db_dirs):
    ok, msg = register.register_face("whatever.jpg", "王" * 51)
    assert not ok and "过长" in msg


def test_register_missing_image(tmp_db_dirs):
    ok, msg = register.register_face("不存在的文件.jpg", "张三")
    assert not ok and "无法读取图片" in msg


def test_register_no_face_detected(tmp_db_dirs, monkeypatch):
    monkeypatch.setattr(register, "FaceDetector", lambda: FakeDetector([]))
    img = make_image(tmp_db_dirs)
    ok, msg = register.register_face(img, "张三")
    assert not ok and "未在照片中检测到人脸" in msg


def test_register_embedding_failure(tmp_db_dirs, monkeypatch):
    class FailingEmbedder:
        def extract(self, img, box):
            return None   # 模拟特征提取失败

    monkeypatch.setattr(register, "FaceDetector", lambda: FakeDetector([(10, 10, 60, 60)]))
    monkeypatch.setattr(register, "FaceEmbedder", lambda: FailingEmbedder())
    img = make_image(tmp_db_dirs)
    ok, msg = register.register_face(img, "张三")
    assert not ok and "特征提取失败" in msg


def test_register_same_name_appends_photo(tmp_db_dirs, monkeypatch):
    monkeypatch.setattr(register, "FaceDetector", lambda: FakeDetector([(10, 10, 60, 60)]))
    monkeypatch.setattr(register, "FaceEmbedder", lambda: FakeEmbedder())
    img = make_image(tmp_db_dirs)

    ok1, _ = register.register_face(img, "张三")
    ok2, msg2 = register.register_face(img, "张三")

    assert ok1 and ok2
    assert "2 张照片" in msg2
    assert len(list((tmp_db_dirs / "emb").glob("*.npy"))) == 1


def test_register_replace_reports_overwrite(tmp_db_dirs, monkeypatch):
    monkeypatch.setattr(register, "FaceDetector", lambda: FakeDetector([(10, 10, 60, 60)]))
    monkeypatch.setattr(register, "FaceEmbedder", lambda: FakeEmbedder())
    img = make_image(tmp_db_dirs)

    register.register_face(img, "张三")
    ok, msg = register.register_face(img, "张三", append=False)

    assert ok and "覆盖" in msg


def test_register_model_download_failure(tmp_db_dirs, monkeypatch):
    class RaisingEmbedder:
        def extract(self, img, box):
            raise RuntimeError("SFace 特征模型下载失败")

    monkeypatch.setattr(register, "FaceDetector", lambda: FakeDetector([(10, 10, 60, 60)]))
    monkeypatch.setattr(register, "FaceEmbedder", lambda: RaisingEmbedder())
    img = make_image(tmp_db_dirs)
    ok, msg = register.register_face(img, "张三")
    assert not ok and "下载失败" in msg
