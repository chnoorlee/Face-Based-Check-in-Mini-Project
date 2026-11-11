# -*- coding: utf-8 -*-
"""
本地存储模块
- FeatureDatabase：人脸特征库（每个学生一个 .npy 特征文件 + 一张人脸照片）
- AttendanceRecorder：签到记录（每天一个 CSV 文件，含 学生姓名/签到时间/是否签到成功 三列）
- find_best_match：特征比对核心函数（找特征库中距离最近的匹配）
"""
import csv
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ATTENDANCE_DIR, EMBEDDINGS_DIR, FACES_DIR
from utils.imutils import ensure_dirs, imwrite_unicode, sanitize_filename

logger = logging.getLogger(__name__)


def find_best_match(embedding, database, tolerance):
    """在特征库中寻找与 embedding 最相似的学生。

    database: {姓名: 特征向量}
    返回 (姓名, 距离)；距离超过 tolerance 视为未匹配，返回 (None, 距离)。
    特征库为空时返回 (None, None)。
    """
    if not database:
        return None, None
    best_name, best_dist = None, float("inf")
    for name, vec in database.items():
        dist = float(np.linalg.norm(np.asarray(embedding) - np.asarray(vec)))
        if dist < best_dist:
            best_name, best_dist = name, dist
    if best_dist <= tolerance:
        return best_name, best_dist
    return None, best_dist


class FeatureDatabase:
    """人脸特征库：data/embeddings/ 下每个学生一个 .npy 文件，文件名即学生姓名。

    同名注册会覆盖旧记录（课程项目规模下姓名即唯一标识）。
    """

    def __init__(self, embeddings_dir=None, faces_dir=None):
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else EMBEDDINGS_DIR
        self.faces_dir = Path(faces_dir) if faces_dir else FACES_DIR
        ensure_dirs(self.embeddings_dir, self.faces_dir)

    def save(self, name, embedding, face_image) -> None:
        """保存特征向量（.npy）与人脸照片（.jpg），同名自动覆盖"""
        fname = sanitize_filename(name)
        np.save(self.embeddings_dir / f"{fname}.npy", embedding)
        imwrite_unicode(self.faces_dir / f"{fname}.jpg", face_image)

    def load_all(self) -> dict:
        """加载全部已注册学生：{姓名: 128 维特征向量}"""
        database = {}
        for p in sorted(self.embeddings_dir.glob("*.npy")):
            database[p.stem] = np.load(p)
        return database

    def exists(self, name) -> bool:
        """学生是否已注册"""
        return (self.embeddings_dir / f"{sanitize_filename(name)}.npy").exists()

    def names(self) -> list:
        """已注册学生姓名列表"""
        return [p.stem for p in sorted(self.embeddings_dir.glob("*.npy"))]


class AttendanceRecorder:
    """签到记录器：每天一个 CSV 文件（attendance_YYYY-MM-DD.csv）

    列为：学生姓名, 签到时间, 是否签到成功
    使用 utf-8-sig 编码，保证 Excel 直接打开中文不乱码。
    """

    CSV_HEADER = ["学生姓名", "签到时间", "是否签到成功"]

    def __init__(self, attendance_dir=None, date=None):
        self.attendance_dir = Path(attendance_dir) if attendance_dir else ATTENDANCE_DIR
        ensure_dirs(self.attendance_dir)
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.path = self.attendance_dir / f"attendance_{self.date}.csv"

    def load_checked_names(self) -> set:
        """读取今天已成功签到的学生名单（用于每日防重复签到）。

        CSV 损坏或编码异常时记录警告并按空记录处理，保证签到流程不中断。
        """
        if not self.path.exists():
            return set()
        try:
            with open(self.path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            logger.warning("读取签到记录失败（%s），按空记录处理：%s", self.path, exc)
            return set()
        checked = set()
        for row in rows[1:]:
            if len(row) >= 3 and row[2] == "是":
                checked.add(row[0])
        return checked

    def record(self, name, success=True) -> dict:
        """追加一条签到记录，返回写入的行内容"""
        # 防 CSV 公式注入：以 = + - @ 开头的姓名会被 Excel 当作公式执行，
        # 统一改写开头字符（与 imutils.sanitize_filename 规则一致，幂等）
        safe_name = name
        while safe_name and safe_name[0] in "=+-@":
            safe_name = "_" + safe_name[1:]
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [safe_name, time_str, "是" if success else "否"]
        new_file = not self.path.exists()
        with open(self.path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.CSV_HEADER)
            writer.writerow(row)
        return dict(zip(self.CSV_HEADER, row))
