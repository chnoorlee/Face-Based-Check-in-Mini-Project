# -*- coding: utf-8 -*-
"""
课堂自动人脸签到系统 - 控制台主入口（课程 3320）

运行：python main.py
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

# 保证从任意目录运行都能找到项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import LOG_DIR, validate_config  # noqa: E402
from utils.imutils import ensure_dirs  # noqa: E402

BANNER = """
=========================================
   课堂自动人脸签到系统（课程 3320）
   YOLO-Face 人脸检测 + 特征比对签到
=========================================
"""

MENU = """  1. 人脸注册    （上传照片，录入特征库）
  2. 实时签到    （摄像头自动识别签到）
  3. 查看信息    （特征库与今日签到记录）
  4. 出勤报告    （统计缺勤名单，无需摄像头）
  5. 删除学生    （从特征库移除）
  6. 退出
=========================================
"""


def setup_logging():
    """日志同时输出到控制台与 data/logs/ 下的日志文件（按大小轮转，1MB × 3 份）"""
    from logging.handlers import RotatingFileHandler
    ensure_dirs(LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(LOG_DIR / f"app_{datetime.now():%Y%m%d}.log",
                                maxBytes=1024 * 1024, backupCount=3, encoding="utf-8"),
        ],
    )


def cmd_register():
    """功能 1：人脸注册"""
    from register import register_face
    image = input("请输入人脸照片路径（可把文件拖进窗口）: ").strip().strip('"')
    name = input("请输入学生姓名: ").strip()
    ok, msg = register_face(image, name)
    print(("✅ " if ok else "❌ ") + msg)


def cmd_checkin():
    """功能 2：实时签到"""
    from checkin import run_console
    run_console()


def cmd_info():
    """功能 3：查看特征库详情与今日签到记录"""
    from utils.storage import AttendanceRecorder, FeatureDatabase
    db = FeatureDatabase()
    infos = db.details()
    recorder = AttendanceRecorder()
    checked = recorder.load_checked_names()
    print("\n已注册学生（{} 人）：".format(len(infos)))
    if not infos:
        print("  （空）")
    for info in infos:
        status = "✅ 今日已签到" if info["name"] in checked else "— 今日未签到"
        print(f"  - {info['name']}（注册于 {info['registered_at']}）{status}")
    print(f"签到文件：{recorder.path}\n")


def cmd_report():
    """功能 4：生成今日出勤报告（统计缺勤名单，无需摄像头）"""
    from utils.roster import Roster
    from utils.storage import AttendanceRecorder, FeatureDatabase
    path = input("应到名单文件路径（直接回车 = 全部已注册学生）: ").strip().strip('"')
    db = FeatureDatabase()
    try:
        roster = Roster.from_file(path) if path else Roster.from_database(db.load_all())
    except (FileNotFoundError, ValueError) as exc:
        print("❌ 读取名单失败：", exc)
        return
    if not len(roster):
        print("❌ 应到名单为空")
        return
    recorder = AttendanceRecorder()
    checked = recorder.load_checked_names()
    report = recorder.write_summary(roster.names, checked)
    present, absent = roster.present(checked), roster.absent(checked)
    print(f"\n出勤：应到 {len(roster)} 人 | 实到 {len(present)} 人 | 缺勤 {len(absent)} 人")
    print(f"✅ 已到（{len(present)}）：{'、'.join(present) or '无'}")
    print(f"❌ 未到（{len(absent)}）：{'、'.join(absent) or '无'}")
    print(f"报告文件：{report}\n")


def cmd_delete():
    """功能 5：删除学生"""
    from utils.storage import FeatureDatabase
    name = input("请输入要删除的学生姓名: ").strip()
    if not name:
        print("❌ 姓名不能为空")
        return
    removed = FeatureDatabase().delete(name)
    print(f"✅ 已删除学生：{name}" if removed else f"❌ 特征库中没有学生：{name}")


def main():
    setup_logging()
    for warning in validate_config():
        print("⚠️ 配置警告：", warning)
    print(BANNER)
    handlers = {"1": cmd_register, "2": cmd_checkin, "3": cmd_info,
                "4": cmd_report, "5": cmd_delete}
    while True:
        print(MENU)
        try:
            choice = input("请选择功能 [1-6]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if choice in handlers:
            handlers[choice]()
        elif choice == "6":
            print("再见！")
            break
        else:
            print("输入无效，请输入 1-6\n")


if __name__ == "__main__":
    main()
