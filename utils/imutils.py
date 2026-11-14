# -*- coding: utf-8 -*-
"""
通用小工具
- 支持中文路径的图片读写（cv2.imread / cv2.imwrite 不支持中文路径，需要绕行）
- 学生姓名到文件名的安全转换
- 目录自动创建
"""
import re
from pathlib import Path

import cv2
import numpy as np

# Windows 文件名中不允许出现的字符
_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\s]+')
# Windows 保留设备名（不能作为文件名，如 "CON.txt" 会失败）
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} \
    | {f"COM{i}" for i in range(1, 10)} \
    | {f"LPT{i}" for i in range(1, 10)}


def csv_safe(value) -> str:
    """把单元格内容转为 Excel 安全的文本。

    以 = + - @ 开头的文本会被 Excel 当作公式执行（CSV 公式注入 / DDE），
    这里统一在前面加一个单引号，使其始终按文本显示。
    """
    text = "" if value is None else str(value)
    if text[:1] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def csv_unescape(value) -> str:
    """还原 csv_safe() 添加的引导单引号，保证读回来的姓名与原始姓名一致。

    否则姓名以 = + - @ 开头的学生，签到记录里的键会多一个单引号，
    与特征库/名单中的姓名对不上，导致每天都被判为缺勤、每日去重也失效。
    """
    text = "" if value is None else str(value)
    if len(text) > 1 and text[0] == "'" and text[1] in ("=", "+", "-", "@"):
        return text[1:]
    return text


def sanitize_filename(name: str) -> str:
    """把学生姓名转换为安全的文件名。

    - 替换掉文件名中的非法字符和空白；
    - 去掉开头的 = + - @ 等字符，防止 CSV 公式注入（Excel 打开时的安全风险）；
    - 避开 Windows 保留设备名（CON、PRN 等）。
    """
    cleaned = _INVALID_FILENAME_CHARS.sub("_", name.strip())
    while cleaned and cleaned[0] in "=+-@":
        cleaned = "_" + cleaned[1:]
    if cleaned.upper() in _RESERVED_NAMES:
        cleaned = "_" + cleaned
    return cleaned if cleaned else "unnamed"


def ensure_dirs(*dirs) -> None:
    """确保目录存在（不存在则创建）"""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def imread_unicode(path):
    """读取图片，支持包含中文的路径。文件不存在或读取失败返回 None。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, TypeError):   # 文件不存在、路径为 None/非法等
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img) -> bool:
    """保存图片，支持包含中文的路径。成功返回 True。"""
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def download_file(dest, urls, hint="", min_size=0) -> bool:
    """按顺序尝试多个地址下载文件到 dest。

    min_size：文件最小字节数。小于该值视为无效内容（例如 Git LFS 指针文本），
    自动删除并换下一个地址。全部失败返回 False（不会残留无效文件）。
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_size:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    import urllib.request
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            size = dest.stat().st_size
            if size < min_size:
                print(f"[模型下载] 下载内容无效（{size} 字节 < {min_size}），换下一个地址...")
                dest.unlink()
                continue
            print(f"[模型下载] {hint or dest.name} 下载完成：{dest}（{size / 1048576:.1f}MB）")
            return True
        except Exception as exc:
            print(f"[模型下载] 下载失败（{url}）：{exc}")
    if dest.exists():
        dest.unlink()
    print(f"[模型下载] {hint or dest.name} 自动下载失败，请手动下载放到 {dest.parent}/ 目录（详见 README）")
    return False
