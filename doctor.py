# -*- coding: utf-8 -*-
"""
环境自检工具（Doctor）

换一台电脑跑不起来时，先运行本工具定位问题：

    python doctor.py

检查项：
  1. Python 版本与 64 位
  2. 依赖库是否可导入（numpy / cv2 / PIL / ultralytics / torch）
  3. 三个模型文件是否存在、大小是否正常
  4. 摄像头是否可用（--no-camera 可跳过）
  5. 中文显示字体是否可用
  6. 数据目录是否可读写
  7. 特征库 / 今日签到记录概况

每项输出 ✅ / ⚠️ / ❌ 与修复建议。
"""
import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import (ATTENDANCE_DIR, CAMERA_INDEX, DATA_DIR,  # noqa: E402
                    EMBEDDINGS_DIR, FACES_DIR, SFACE_MODEL, YUNET_MODEL,
                    YOLO_FACE_MODEL, validate_config)

OK, WARN, FAIL = "✅", "⚠️", "❌"
results = []


def check(label, fn):
    """执行一项检查并记录结果"""
    try:
        ok, detail, hint = fn()
    except Exception as exc:  # 检查本身出错也不应中断自检
        ok, detail, hint = "fail", f"检查异常：{exc}", ""
    mark = {True: OK, "warn": WARN, "fail": FAIL}.get(ok, WARN)
    results.append((mark, label, detail, hint))
    print(f"{mark} {label}：{detail}")
    if hint:
        print(f"     ↳ {hint}")
    return ok


# ---------------- 各项检查 ----------------
def check_python():
    v = sys.version_info
    bits = struct.calcsize("P") * 8
    detail = f"Python {v.major}.{v.minor}.{v.micro}（{bits} 位）"
    if bits != 64:
        return "fail", detail + "，需要 64 位 Python", "请安装 64 位 Python 3.9+"
    if v < (3, 9):
        return "fail", detail + "，版本过低", "需要 Python 3.9 及以上（推荐 3.10~3.12）"
    if v >= (3, 13):
        return "warn", detail, "3.13+ 未充分测试，建议使用 3.10~3.12"
    return True, detail, ""


def check_deps():
    missing, broken = [], []
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python"),
                     ("PIL", "Pillow"), ("ultralytics", "ultralytics"), ("torch", "torch")):
        try:
            __import__(mod)
        except ImportError as exc:
            (broken if "No module named" not in str(exc) else missing).append(pkg)
    if missing or broken:
        return "fail", f"缺少依赖：{'、'.join(missing + broken)}", "执行 pip install -r requirements.txt"
    import cv2 as _cv2
    return True, f"全部可导入（opencv {_cv2.__version__}）", ""


def check_models():
    specs = [
        (YOLO_FACE_MODEL, "YOLO-Face", 1 * 1024 * 1024),
        (SFACE_MODEL, "SFace", 1 * 1024 * 1024),
        (YUNET_MODEL, "YuNet", 64 * 1024),
    ]
    missing, tiny = [], []
    for path, name, min_size in specs:
        p = Path(path)
        if not p.exists():
            missing.append(name)
        elif p.stat().st_size < min_size:
            tiny.append(f"{name}({p.stat().st_size}B)")
    if missing or tiny:
        parts = []
        if missing:
            parts.append(f"缺失：{'、'.join(missing)}")
        if tiny:
            parts.append(f"文件异常：{'、'.join(tiny)}")
        return "warn", "；".join(parts), "首次运行会自动下载；也可手动下载到 models/（地址见 config.py）"
    return True, f"3 个模型就绪（{sum(Path(p).stat().st_size for p, _, _ in specs) / 1048576:.1f}MB）", ""


def check_camera():
    try:
        import cv2
    except ImportError:
        return "fail", "opencv 未安装", "pip install -r requirements.txt"
    from utils.camera import diagnose_frame, warmup
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        return "fail", f"无法打开摄像头 {CAMERA_INDEX}", \
            "检查摄像头占用/权限，或在 config.py 修改 CAMERA_INDEX（外接摄像头试 1）"
    # 按真实时间预热（实测自动曝光需 1~2 秒），再判断画面是否可用
    _, frame = warmup(cap)
    cap.release()
    if frame is None:
        return "warn", "摄像头已打开但读不到画面", "稍后重试，或检查摄像头是否被其他程序占用"
    healthy, message = diagnose_frame(frame)
    if not healthy:
        return "fail", f"{frame.shape[1]}x{frame.shape[0]}，但{message}", "按上面的提示处理后再运行"
    return True, f"摄像头 {CAMERA_INDEX} 可用（{frame.shape[1]}x{frame.shape[0]}，{message}）", ""


def check_font():
    faces = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
             "/System/Library/Fonts/PingFang.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for f in faces:
        if Path(f).exists():
            return True, f"中文标签字体可用（{Path(f).name}）", ""
    return "warn", "未找到中文字体", "姓名标签会退化为英文绘制，不影响识别与签到"


def check_data_dirs():
    for d in (DATA_DIR, FACES_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=d, delete=True):
                pass
        except OSError as exc:
            return "fail", f"目录不可写：{d}", f"检查权限或磁盘空间（{exc}）"
    return True, "数据目录可读写", ""


def check_database():
    from utils.storage import AttendanceRecorder, FeatureDatabase
    db = FeatureDatabase()
    names = db.names()
    recorder = AttendanceRecorder()
    checked = recorder.load_checked_names()
    detail = f"已注册 {len(names)} 人；今日已签到 {len(checked)} 人"
    if not names:
        return "warn", detail, "还没有学生，先运行 python register.py -i 照片.jpg -n 姓名"
    return True, detail, ""


def check_config():
    warnings = validate_config()
    if warnings:
        return "warn", "；".join(warnings), "修改 config.py 中的对应配置项"
    return True, "配置合法", ""


def main():  # pragma: no cover - 命令行入口
    parser = argparse.ArgumentParser(description="课堂人脸签到系统 - 环境自检")
    parser.add_argument("--no-camera", action="store_true", help="跳过摄像头检查")
    args = parser.parse_args()

    print("=" * 52)
    print("  课堂自动人脸签到系统 · 环境自检")
    print("=" * 52)

    check("Python 版本", check_python)
    check("依赖库", check_deps)
    check("模型文件", check_models)
    check("配置文件", check_config)
    check("数据目录", check_data_dirs)
    check("中文显示", check_font)
    check("特征库", check_database)
    if not args.no_camera:
        check("摄像头", check_camera)

    fails = sum(1 for m, *_ in results if m == FAIL)
    warns = sum(1 for m, *_ in results if m == WARN)
    print("-" * 52)
    if fails:
        print(f"结论：{fails} 项失败、{warns} 项警告 —— 请按上面的建议修复后再运行")
        return 1
    if warns:
        print(f"结论：{warns} 项警告，其余正常 —— 通常仍可运行")
        return 0
    print("结论：全部通过，可以正常运行 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
