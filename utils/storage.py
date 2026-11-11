# -*- coding: utf-8 -*-
"""
本地存储模块
- FeatureDatabase：人脸特征库（每个学生一个 .npy 特征文件 + 一张人脸照片 + 姓名索引）
- AttendanceRecorder：签到记录（按天/节次一个 CSV，含 学生姓名/签到时间/是否签到成功）
- write_summary：四态出勤报告（已到/迟到/请假/缺勤）
- find_best_match：特征比对核心函数
"""
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ATTENDANCE_DIR, EMBEDDINGS_DIR, FACES_DIR
from utils.imutils import (csv_safe, ensure_dirs, imwrite_unicode,
                           sanitize_filename)

logger = logging.getLogger(__name__)


def find_best_match(embedding, database, tolerance):
    """在特征库中寻找与 embedding 最相似的学生。

    database: {姓名: 特征向量}
    返回 (姓名, 距离)；距离超过 tolerance 视为未匹配，返回 (None, 距离)。
    特征库为空、或特征维度不一致时返回 (None, None)。
    """
    if not database:
        return None, None
    probe = np.asarray(embedding).ravel()
    best_name, best_dist = None, float("inf")
    for name, vec in database.items():
        vec = np.asarray(vec).ravel()
        if vec.shape != probe.shape:      # 维度异常的历史特征，跳过而不是崩溃
            continue
        dist = float(np.linalg.norm(probe - vec))
        if dist < best_dist:
            best_name, best_dist = name, dist
    if best_name is None:
        return None, None
    if best_dist <= tolerance:
        return best_name, best_dist
    return None, best_dist


class FeatureDatabase:
    """人脸特征库。

    每个学生：
      - data/embeddings/姓名.npy   128 维特征向量
      - data/faces/姓名.jpg        注册时的人脸照片
      - data/embeddings/_index.json 记录「显示姓名 → 文件名」映射与注册时间

    索引文件解决了两个问题：
      1. 姓名含特殊字符时会被改写成文件名（"张 三" -> "张_三"），
         索引保留原始姓名，报告与名单比对仍用原名；
      2. 两个不同姓名若改写成同一个文件名，注册时会被拒绝，避免静默覆盖掉另一个学生。
    """

    INDEX_NAME = "_index.json"

    def __init__(self, embeddings_dir=None, faces_dir=None):
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else EMBEDDINGS_DIR
        self.faces_dir = Path(faces_dir) if faces_dir else FACES_DIR
        ensure_dirs(self.embeddings_dir, self.faces_dir)
        self._index_path = self.embeddings_dir / self.INDEX_NAME

    # ---------- 索引 ----------
    def _load_index(self) -> dict:
        try:
            if self._index_path.exists():
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if isinstance(v, dict)}
                logger.warning("特征库索引格式异常，将按文件名重建：%s", self._index_path)
        except (OSError, ValueError) as exc:
            logger.warning("特征库索引读取失败，将按文件名重建：%s", exc)
        return {}

    def _write_index(self, index: dict) -> None:
        tmp = self._index_path.with_name(self.INDEX_NAME + ".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._index_path)

    @staticmethod
    def _stem_owner(stem, index):
        for name, meta in index.items():
            if meta.get("stem") == stem:
                return name
        return None

    # ---------- 读写 ----------
    def save(self, name, embedding, face_image):
        """保存特征向量与人脸照片。返回 (是否成功, 提示信息)。"""
        fname = sanitize_filename(name)
        index = self._load_index()

        owner = self._stem_owner(fname, index)
        if owner is not None and owner != name:
            return False, (f"姓名冲突：「{name}」与已注册的「{owner}」会写入同一个文件 "
                           f"{fname}.npy，为避免覆盖请不要使用仅特殊字符不同的姓名")

        vec_path = self.embeddings_dir / f"{fname}.npy"
        tmp_path = self.embeddings_dir / f".{fname}.tmp.npy"
        try:
            np.save(tmp_path, np.asarray(embedding))     # 先写临时文件，再原子替换
            os.replace(tmp_path, vec_path)
        except OSError as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return False, f"保存人脸特征失败：{exc}"

        photo_ok = imwrite_unicode(self.faces_dir / f"{fname}.jpg", face_image)
        if not photo_ok:
            logger.warning("人脸照片保存失败（特征已入库）：%s", name)

        index[name] = {"stem": fname,
                       "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        try:
            self._write_index(index)
        except OSError as exc:
            logger.warning("更新特征库索引失败：%s", exc)

        return True, "" if photo_ok else "（人脸照片保存失败，特征已入库）"

    def load_all(self) -> dict:
        """加载全部已注册学生：{显示姓名: 特征向量}。

        单个特征文件损坏时跳过并记录警告，不影响其他学生使用。
        """
        index = self._load_index()
        stem_to_name = {meta.get("stem"): name for name, meta in index.items() if meta.get("stem")}
        database = {}
        for p in sorted(self.embeddings_dir.glob("*.npy")):
            name = stem_to_name.get(p.stem, p.stem)
            try:
                vec = np.asarray(np.load(p))
            except (OSError, ValueError) as exc:
                logger.warning("跳过无法读取的特征文件 %s：%s", p.name, exc)
                continue
            if vec.ndim != 1 or vec.size == 0:
                logger.warning("跳过维度异常的特征文件 %s（shape=%s）", p.name, vec.shape)
                continue
            database[name] = vec
        return database

    def exists(self, name) -> bool:
        """学生是否已注册"""
        if name in self._load_index():
            return True
        return (self.embeddings_dir / f"{sanitize_filename(name)}.npy").exists()

    def names(self) -> list:
        """已注册学生姓名列表（显示姓名）"""
        index = self._load_index()
        stem_to_name = {meta.get("stem"): name for name, meta in index.items() if meta.get("stem")}
        return [stem_to_name.get(p.stem, p.stem) for p in sorted(self.embeddings_dir.glob("*.npy"))]

    def delete(self, name) -> bool:
        """删除学生（特征向量 + 人脸照片 + 索引）。返回是否有记录被删除。"""
        index = self._load_index()
        stem = index.get(name, {}).get("stem") or sanitize_filename(name)
        removed = False
        for p in (self.embeddings_dir / f"{stem}.npy", self.faces_dir / f"{stem}.jpg"):
            try:
                if p.exists():
                    p.unlink()
                    removed = True
            except OSError as exc:
                logger.warning("删除文件失败 %s：%s", p, exc)
        if name in index:
            index.pop(name, None)
            try:
                self._write_index(index)
            except OSError as exc:
                logger.warning("更新特征库索引失败：%s", exc)
            removed = True
        return removed

    def details(self) -> list:
        """已注册学生详情：[{name, registered_at}]"""
        index = self._load_index()
        infos = []
        for p in sorted(self.embeddings_dir.glob("*.npy")):
            meta = None
            for name, m in index.items():
                if m.get("stem") == p.stem:
                    meta = (name, m)
                    break
            if meta is None:
                meta = (p.stem, {})
            registered = meta[1].get("registered_at") or \
                datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            infos.append({"name": meta[0], "registered_at": registered})
        return infos


class AttendanceRecorder:
    """签到记录器：每天一个 CSV 文件（attendance_YYYY-MM-DD.csv）

    列为：学生姓名, 签到时间, 是否签到成功
    使用 utf-8-sig 编码，保证 Excel 直接打开中文不乱码。

    节次（session）：同一天有多节课时传入节次名称，记录写入
    attendance_YYYY-MM-DD_节次.csv，每日去重也随之按节次独立计算。
    """

    CSV_HEADER = ["学生姓名", "签到时间", "是否签到成功"]

    def __init__(self, attendance_dir=None, date=None, session=None):
        self.attendance_dir = Path(attendance_dir) if attendance_dir else ATTENDANCE_DIR
        ensure_dirs(self.attendance_dir)
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.session = (session or "").strip() or None
        self._suffix = f"_{sanitize_filename(self.session)}" if self.session else ""
        self.path = self.attendance_dir / f"attendance_{self.date}{self._suffix}.csv"
        self.last_read_error = None   # 最近一次读取失败原因（None 表示读取正常）

    def load_checked_names(self) -> set:
        """读取今天已成功签到的学生名单（用于每日防重复签到）。"""
        return set(self.load_records().keys())

    def load_records(self) -> dict:
        """读取今天已成功签到的明细：{姓名: 签到时间}。

        文件不存在视为「今天还没人签到」；文件存在但读取失败时
        记录到 self.last_read_error 并返回空字典（调用方据此中止生成报告，
        避免把所有人错写成缺勤）。
        """
        self.last_read_error = None
        if not self.path.exists():
            return {}
        try:
            with open(self.path, encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            self.last_read_error = str(exc)
            logger.warning("读取签到记录失败（%s）：%s", self.path, exc)
            return {}
        records = {}
        for row in rows[1:]:
            if len(row) >= 3 and row[2] == "是":
                records.setdefault(row[0], row[1])
        return records

    def write_summary(self, roster_names, checked_names, leave_names=(), late_after=None):
        """写出勤报告 CSV：应到名单中每人一行，状态为 已到 / 迟到 / 请假 / 缺勤。

        文件名：attendance_YYYY-MM-DD[_节次]_report.csv
        返回报告路径；若签到记录读取失败则返回 None（不覆盖已有报告）。
        """
        records = self.load_records()
        if self.last_read_error:
            logger.warning("签到记录读取失败，已跳过生成出勤报告：%s", self.last_read_error)
            return None

        leave = set(leave_names or ())
        path = self.report_path()
        tmp = path.with_name(path.stem + ".tmp" + path.suffix)
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["学生姓名", "状态", "签到时间"])
            for name in roster_names:
                time_str = records.get(name, "")
                if name in leave:
                    writer.writerow([csv_safe(name), "请假", ""])
                elif name in checked_names:
                    status = "迟到" if is_late(time_str, late_after) else "已到"
                    writer.writerow([csv_safe(name), status, csv_safe(time_str)])
                else:
                    writer.writerow([csv_safe(name), "缺勤", ""])
        os.replace(tmp, path)   # 原子替换，避免中途崩溃截断上一份报告
        return path

    def report_path(self) -> Path:
        """出勤报告文件路径（与明细表同目录、同日期/节次）"""
        return self.attendance_dir / f"attendance_{self.date}{self._suffix}_report.csv"

    def record(self, name, success=True) -> dict:
        """追加一条签到记录，返回写入的行内容。写入失败会抛出 OSError。"""
        safe_name = csv_safe(name)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [safe_name, time_str, "是" if success else "否"]

        # 文件不存在、或存在但为空（上次崩溃留下）都视为新文件，需要补表头
        new_file = (not self.path.exists()) or self.path.stat().st_size == 0
        with open(self.path, "a", encoding="utf-8-sig", newline="") as f:
            # 上一行若因崩溃缺少换行，先补一个，避免两条记录粘连成坏行
            if not new_file:
                with open(self.path, "rb") as rf:
                    rf.seek(-1, os.SEEK_END)
                    if rf.read(1) not in (b"\n", b"\r"):
                        f.write("\n")
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.CSV_HEADER)
            writer.writerow(row)
        return dict(zip(self.CSV_HEADER, row))


def is_late(time_str, late_after) -> bool:
    """判断签到时间字符串（YYYY-MM-DD HH:MM:SS）是否晚于迟到判定时刻。

    late_after 为 None 或时间字符串无法解析时返回 False（不判迟到）。
    """
    if late_after is None or not time_str:
        return False
    try:
        stamp = datetime.strptime(time_str.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return stamp.time() > late_after
