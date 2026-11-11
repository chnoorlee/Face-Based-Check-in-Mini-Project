# -*- coding: utf-8 -*-
"""utils.imutils 单元测试：中文路径读写、文件名清洗、模型下载"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.imutils import (download_file, imread_unicode, imwrite_unicode,
                           sanitize_filename)


def _img():
    return np.zeros((32, 32, 3), dtype=np.uint8)


class TestSanitizeFilename:
    def test_keeps_chinese_name(self):
        assert sanitize_filename("张三") == "张三"

    def test_strips_illegal_chars(self):
        out = sanitize_filename('张/三:*?"<>|')
        assert "/" not in out and ":" not in out and "*" not in out

    def test_replaces_whitespace(self):
        assert sanitize_filename("张 三") == "张_三"

    def test_leading_formula_chars(self):
        # CSV 公式注入只与第一个字符有关：开头字符换成 _ 后即为安全文件名
        assert sanitize_filename("=cmd") == "_cmd"
        assert sanitize_filename("+x") == "_x"
        assert sanitize_filename("@y") == "_y"
        assert sanitize_filename("-=x") == "_=x"

    def test_reserved_device_names(self):
        assert sanitize_filename("CON") == "_CON"
        assert sanitize_filename("com1") == "_com1"

    def test_empty_returns_unnamed(self):
        assert sanitize_filename("   ") == "unnamed"


class TestImageIO:
    def test_roundtrip_with_chinese_path(self, tmp_path):
        p = tmp_path / "测试 照片.jpg"
        assert imwrite_unicode(p, _img())
        loaded = imread_unicode(p)
        assert loaded is not None and loaded.shape == (32, 32, 3)

    def test_read_missing_file_returns_none(self, tmp_path):
        assert imread_unicode(tmp_path / "不存在.jpg") is None

    def test_read_directory_returns_none(self, tmp_path):
        # 路径是目录而非文件时应返回 None 而不是抛异常
        assert imread_unicode(tmp_path) is None


class TestDownloadFile:
    def test_download_from_local_file_url(self, tmp_path):
        # 用 file:// URL 离线模拟真实下载
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 100)
        dest = tmp_path / "sub" / "dst.bin"
        assert download_file(dest, [src.as_uri()], min_size=50)
        assert dest.read_bytes() == b"x" * 100

    def test_already_downloaded_is_kept(self, tmp_path):
        dest = tmp_path / "dst.bin"
        dest.write_bytes(b"ok" * 100)
        assert download_file(dest, ["http://invalid.example/x"], min_size=10)
        assert dest.read_bytes() == b"ok" * 100

    def test_too_small_content_is_rejected(self, tmp_path):
        src = tmp_path / "small.bin"
        src.write_bytes(b"tiny")   # 模拟 Git LFS 指针文件
        dest = tmp_path / "dst.bin"
        assert not download_file(dest, [src.as_uri()], min_size=100)
        assert not dest.exists()   # 无效文件应被清除

    def test_all_urls_fail_returns_false(self, tmp_path):
        dest = tmp_path / "dst.bin"
        assert not download_file(dest, ["file:///C:/no/such/file.bin",
                                        "http://127.0.0.1:1/x"], min_size=1)
        assert not dest.exists()
