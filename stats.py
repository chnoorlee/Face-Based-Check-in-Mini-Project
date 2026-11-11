# -*- coding: utf-8 -*-
"""
出勤统计报表模块

把 data/attendance/ 下所有历史签到记录汇总成一张「每人每学期」的出勤统计表：
    学生姓名 / 应到次数 / 实到次数 / 迟到次数 / 请假次数 / 缺勤次数 / 出勤率

统计口径：
  - 一个签到明细文件 = 一次课（节次）；
  - 优先读取对应的 _report.csv（已有已到/迟到/请假/缺勤四态）；
  - 没有报告文件时，按明细 + config 中的上课时间推算是否迟到；
  - 请假信息只存在于报告文件中，故无报告的历史课次无法统计请假。

用法：
    python stats.py                       # 控制台输出 + 写出勤统计.csv
    python stats.py --roster 名单.txt      # 只统计名单内学生
    python stats.py --from 2026-09-01 --to 2026-12-31
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import ATTENDANCE_DIR  # noqa: E402
from utils.roster import Roster  # noqa: E402
from utils.storage import is_late  # noqa: E402

STATUSES = ("已到", "迟到", "请假", "缺勤")


def discover_sessions(attendance_dir):
    """扫描签到文件，返回 {课次键: {"detail": Path, "report": Path|None}}。

    课次键形如 "2026-11-11" 或 "2026-11-11_第1节"（同一天多节课）。
    """
    sessions = {}
    for p in sorted(Path(attendance_dir).glob("attendance_*.csv")):
        stem = p.stem  # attendance_2026-11-11[_节次][_report]
        if not stem.startswith("attendance_"):
            continue
        key = stem[len("attendance_"):]
        if key.endswith("_report"):
            key = key[: -len("_report")]
            sessions.setdefault(key, {"detail": None, "report": None})["report"] = p
        else:
            sessions.setdefault(key, {"detail": None, "report": None})["detail"] = p
    return {k: v for k, v in sessions.items() if v["detail"] or v["report"]}


def read_status_map(session, late_after=None):
    """读取一次课的「姓名 -> 状态」映射。

    优先用报告文件（含请假/迟到）；没有报告时用签到明细推算。
    """
    status = {}
    report = session.get("report")
    if report is not None:
        try:
            with open(report, encoding="utf-8-sig", newline="") as f:
                for row in list(csv.reader(f))[1:]:
                    if len(row) >= 2 and row[1] in STATUSES:
                        status[row[0]] = row[1]
        except (OSError, csv.Error, UnicodeDecodeError):
            pass
    if not status:
        detail = session.get("detail")
        if detail is not None:
            try:
                with open(detail, encoding="utf-8-sig", newline="") as f:
                    for row in list(csv.reader(f))[1:]:
                        if len(row) >= 3 and row[2] == "是":
                            status[row[0]] = "迟到" if is_late(row[1], late_after) else "已到"
            except (OSError, csv.Error, UnicodeDecodeError):
                pass
    return status


def summarize(attendance_dir, names=None, date_from=None, date_to=None, late_after=None):
    """汇总出勤统计。返回 (行列表, 课次列表)。

    每行：{姓名, 应到, 实到, 迟到, 请假, 缺勤, 出勤率}
    """
    sessions = discover_sessions(attendance_dir)
    keys = []
    for key in sessions:
        date_part = key[:10]
        if date_from and date_part < date_from:
            continue
        if date_to and date_part > date_to:
            continue
        keys.append(key)
    keys.sort()

    stats = {}
    for key in keys:
        for name, st in read_status_map(sessions[key], late_after).items():
            row = stats.setdefault(name, dict.fromkeys(STATUSES, 0))
            row[st] = row.get(st, 0) + 1

    target = list(names) if names else sorted(stats)
    rows = []
    for name in target:
        counts = stats.get(name, dict.fromkeys(STATUSES, 0))
        attended = counts["已到"] + counts["迟到"]
        leave = counts["请假"]
        total = len(keys)
        # 缺勤 = 应到课次 − 实到 − 请假（明细里没出现的课次即为缺勤）
        absent = max(total - attended - leave, 0)
        rows.append({
            "姓名": name,
            "应到": total,
            "实到": attended,
            "迟到": counts["迟到"],
            "请假": leave,
            "缺勤": absent,
            "出勤率": f"{(attended / total * 100):.1f}%" if total else "-",
        })
    return rows, keys


def write_stats_csv(rows, out_path):
    """把统计结果写成 CSV（utf-8-sig，Excel 可直接打开）"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["学生姓名", "应到次数", "实到次数", "迟到次数", "请假次数", "缺勤次数", "出勤率"])
        for r in rows:
            writer.writerow([r["姓名"], r["应到"], r["实到"], r["迟到"], r["请假"], r["缺勤"], r["出勤率"]])
    return out_path


def print_table(rows, keys):
    """控制台打印统计表"""
    print(f"统计课次：{len(keys)} 次" + (f"（{keys[0]} ~ {keys[-1]}）" if keys else ""))
    if not rows:
        print("没有可统计的签到记录。")
        return
    header = f"{'学生姓名':<10}{'应到':>5}{'实到':>5}{'迟到':>5}{'请假':>5}{'缺勤':>5}{'出勤率':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = r["姓名"]
        # 中文按显示宽度对齐（粗略：每字 2 列）
        pad = 10 - sum(2 if ord(c) > 127 else 1 for c in name)
        print(f"{name}{' ' * max(pad, 0)}{r['应到']:>5}{r['实到']:>5}{r['迟到']:>5}"
              f"{r['请假']:>5}{r['缺勤']:>5}{r['出勤率']:>9}")


def main():  # pragma: no cover - 命令行入口
    parser = argparse.ArgumentParser(description="课堂人脸签到系统 - 出勤统计报表")
    parser.add_argument("--dir", "-d", default=str(ATTENDANCE_DIR), help="签到记录目录")
    parser.add_argument("--roster", "-r", help="只统计名单内的学生（.txt / .csv）")
    parser.add_argument("--from", dest="date_from", help="起始日期 YYYY-MM-DD（含）")
    parser.add_argument("--to", dest="date_to", help="结束日期 YYYY-MM-DD（含）")
    parser.add_argument("--out", "-o", help="统计表输出路径（默认 <目录>/出勤统计.csv）")
    args = parser.parse_args()

    names = None
    if args.roster:
        try:
            names = Roster.from_file(args.roster).names
        except (FileNotFoundError, ValueError) as exc:
            print("❌ 读取名单失败：", exc)
            return 1

    from config import late_after as config_late_after
    rows, keys = summarize(Path(args.dir), names, args.date_from, args.date_to,
                           config_late_after())
    print_table(rows, keys)
    out = Path(args.out) if args.out else Path(args.dir) / "出勤统计.csv"
    path = write_stats_csv(rows, out)
    print(f"\n统计表已保存：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
