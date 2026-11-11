# -*- coding: utf-8 -*-
"""
一键启动（双击即用）

打开就进入摄像头签到：不需要菜单、不需要按任何键。
    人站到摄像头前 → 自动框出人脸 → 识别成功 → 右侧名单对应条目自动打勾

用法：
    双击「启动签到.bat」，或命令行 python start.py

首次使用若特征库为空，会引导先注册一张人脸。
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK；pythonw 无控制台，reconfigure 失败也无妨
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import tkinter as tk  # noqa: E402
from tkinter import messagebox  # noqa: E402

from config import LOG_DIR, validate_config  # noqa: E402
from utils.imutils import ensure_dirs  # noqa: E402


def setup_logging():
    """pythonw 启动没有控制台，日志写文件便于排查"""
    from logging.handlers import RotatingFileHandler
    handlers = []
    try:
        ensure_dirs(LOG_DIR)
        handlers.append(RotatingFileHandler(
            LOG_DIR / f"app_{datetime.now():%Y%m%d}.log",
            maxBytes=1024 * 1024, backupCount=3, encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=handlers or [logging.NullHandler()])


def ensure_registered(root) -> bool:
    """没有已注册学生时引导注册；返回是否可以进入签到"""
    from gui import RegisterDialog
    from utils.storage import FeatureDatabase
    if FeatureDatabase().names():
        return True
    if not messagebox.askyesno(
            "还没有注册学生",
            "特征库里还没有任何学生，无法识别。\n\n现在注册一张人脸吗？"):
        return False
    root.deiconify()
    dialog = RegisterDialog(root, on_success=lambda: None)
    root.wait_window(dialog)
    root.withdraw()
    return bool(FeatureDatabase().names())


def run():
    """启动一键签到窗口"""
    root = tk.Tk()
    root.withdraw()          # 隐藏空的根窗口，只显示签到窗口

    for warning in validate_config():
        print("⚠️ 配置警告：", warning)

    try:
        if not ensure_registered(root):
            messagebox.showinfo(
                "提示",
                "没有已注册学生，暂时无法签到。\n\n"
                "需要完整功能（注册/统计/环境自检）请运行 python main.py")
            return 0

        from gui import CheckinWindow
        CheckinWindow(root, standalone=True)
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


def main():
    setup_logging()
    try:
        return run()
    except Exception as exc:      # pythonw 下没有控制台，必须弹窗告知
        logging.exception("一键启动失败")
        try:
            err = tk.Tk()
            err.withdraw()
            messagebox.showerror("启动失败",
                                 f"{exc}\n\n可先运行 python doctor.py 检查环境，"
                                 f"或运行 python main.py 使用完整菜单。\n\n"
                                 f"日志：{LOG_DIR}")
            err.destroy()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
