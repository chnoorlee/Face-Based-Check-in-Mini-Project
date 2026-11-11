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

from config import LOG_DIR  # noqa: E402
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
  4. 退出
=========================================
"""


def setup_logging():
    """日志同时输出到控制台与 data/logs/ 下的日志文件"""
    ensure_dirs(LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / f"app_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
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
    """功能 3：查看特征库与今日签到记录"""
    from utils.storage import AttendanceRecorder, FeatureDatabase
    db = FeatureDatabase()
    names = db.names()
    print(f"\n已注册学生（{len(names)} 人）：{'、'.join(names) or '（空）'}")
    recorder = AttendanceRecorder()
    checked = recorder.load_checked_names()
    print(f"今日已签到（{len(checked)} 人）：{'、'.join(sorted(checked)) or '（暂无）'}")
    print(f"签到文件：{recorder.path}\n")


def main():
    setup_logging()
    print(BANNER)
    handlers = {"1": cmd_register, "2": cmd_checkin, "3": cmd_info}
    while True:
        print(MENU)
        choice = input("请选择功能 [1-4]: ").strip()
        if choice in handlers:
            handlers[choice]()
        elif choice == "4":
            print("再见！")
            break
        else:
            print("输入无效，请输入 1-4\n")


if __name__ == "__main__":
    main()
