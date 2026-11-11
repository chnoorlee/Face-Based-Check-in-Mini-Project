# -*- coding: utf-8 -*-
"""
应到名单与出勤统计模块

Roster（应到名单）负责回答三个问题：
  1. 今天这节课应该到哪些人？
  2. 其中哪些人已经签到（打勾）？
  3. 哪些人没来（缺勤）？

名单来源有两种：
  - 默认：特征库中所有已注册学生（最常用，无需额外文件）
  - 指定文件：--roster 名单.txt（每行一个姓名）或 名单.csv（第一列，可含表头）
"""
from pathlib import Path


class Roster:
    """应到学生名单"""

    def __init__(self, names):
        # 去空白、去重、保持原顺序
        seen, cleaned = set(), []
        for n in names:
            n = (n or "").strip()
            if n and n not in seen:
                seen.add(n)
                cleaned.append(n)
        self.names = cleaned

    # ---------- 构造 ----------
    @classmethod
    def from_database(cls, database):
        """默认名单：特征库中所有已注册学生"""
        return cls(list(database.keys()))

    @classmethod
    def from_file(cls, path):
        """从文件读取名单：.csv 取第一列（跳过表头），其余格式每行一个姓名。

        支持 UTF-8（含 BOM）与 GBK 编码，兼容 Excel 导出的名单。
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"名单文件不存在：{path}")
        text = None
        for enc in ("utf-8-sig", "gbk"):
            try:
                text = p.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError(f"名单文件编码无法识别（请另存为 UTF-8）：{path}")

        names = []
        is_csv = p.suffix.lower() == ".csv"
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = line.split(",")[0].strip().strip('"') if is_csv else line
            if not name:
                continue
            # CSV 第一行可能是表头
            if is_csv and i == 0 and name in ("学生姓名", "姓名", "name", "Name"):
                continue
            names.append(name)
        return cls(names)

    # ---------- 统计 ----------
    def __len__(self):
        return len(self.names)

    def present(self, checked):
        """已签到（已到）的学生名单"""
        return [n for n in self.names if n in checked]

    def absent(self, checked):
        """未签到（缺勤）的学生名单"""
        return [n for n in self.names if n not in checked]

    def unregistered(self, database):
        """名单中尚未注册人脸、因而无法被识别的学生"""
        return [n for n in self.names if n not in database]

    def progress_text(self, checked):
        """进度文字，如 "3/8" """
        return f"{len(self.present(checked))}/{len(self)}"
