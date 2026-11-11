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
import csv
import io
from pathlib import Path


class Roster:
    """应到学生名单（含请假标记）"""

    LEAVE_MARKERS = ("请假", "leave", "Leave", "LEAVE")

    def __init__(self, names, leave=()):
        # 去空白、去重、保持原顺序
        seen, cleaned = set(), []
        for n in names:
            n = (n or "").strip()
            if n and n not in seen:
                seen.add(n)
                cleaned.append(n)
        self.names = cleaned
        # 请假名单：只保留确实在应到名单中的人
        leave_set = {(n or "").strip() for n in leave}
        self.leave = {n for n in self.names if n in leave_set}

    # ---------- 构造 ----------
    @classmethod
    def from_database(cls, database, leave=()):
        """默认名单：特征库中所有已注册学生"""
        return cls(list(database.keys()), leave=leave)

    @classmethod
    def from_file(cls, path, leave=()):
        """从文件读取名单：.csv 取第一列（跳过表头），其余格式每行一个姓名。

        CSV 中若同一行出现「请假」字样，该学生自动登记为请假。
        支持 UTF-8（含 BOM）与 GBK 编码，兼容 Excel 导出的名单。
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"名单文件不存在或不是文件：{path}")

        text, used_encoding = None, None
        for enc in ("utf-8-sig", "gbk"):
            try:
                text = p.read_text(encoding=enc)
                used_encoding = enc
                break
            except UnicodeDecodeError:
                continue
            except OSError as exc:      # 权限、被占用、是目录等
                raise ValueError(f"无法读取名单文件（{exc}）：{path}") from exc
        if text is None:
            raise ValueError(f"名单文件编码无法识别（请另存为 UTF-8）：{path}")
        if used_encoding == "gbk":
            print(f"⚠️ 名单文件按 GBK 解码（{p.name}）；若中文出现乱码，请用 Excel 另存为 UTF-8 CSV")

        names, from_file_leave = [], set()
        is_csv = p.suffix.lower() == ".csv"
        if is_csv:
            # 用标准 csv 解析，正确处理带引号、含逗号的字段
            try:
                rows = list(csv.reader(io.StringIO(text)))
            except csv.Error as exc:
                raise ValueError(f"名单 CSV 格式错误：{exc}") from exc
            first_data_row = True
            for cells in rows:
                if not cells:
                    continue
                cells = [c.strip() for c in cells]
                name = cells[0]
                if not name or name.startswith("#"):
                    continue
                # 只把「第一行有效数据」当作表头判断，避免注释/空行导致表头被当成学生
                if first_data_row and name in ("学生姓名", "姓名", "name", "Name"):
                    first_data_row = False
                    continue
                first_data_row = False
                # 任意一列含「请假」即视为请假登记
                if any(m in cell for cell in cells[1:] for m in cls.LEAVE_MARKERS):
                    from_file_leave.add(name)
                names.append(name)
        else:
            for line in text.splitlines():
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                names.append(name)
        return cls(names, leave=set(leave) | from_file_leave)

    # ---------- 统计 ----------
    def __len__(self):
        return len(self.names)

    def present(self, checked):
        """已签到（已到，含迟到）的学生名单"""
        return [n for n in self.names if n in checked]

    def on_leave(self):
        """登记请假的学生名单"""
        return [n for n in self.names if n in self.leave]

    def absent(self, checked):
        """缺勤名单：应到名单中既没签到、也没请假的人"""
        return [n for n in self.names if n not in checked and n not in self.leave]

    def unregistered(self, database):
        """名单中尚未注册人脸、因而无法被识别的学生"""
        return [n for n in self.names if n not in database]

    def progress_text(self, checked):
        """进度文字，如 "3/8" """
        return f"{len(self.present(checked))}/{len(self)}"
