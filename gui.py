# -*- coding: utf-8 -*-
"""
可选 tkinter 图形界面（简易版）

运行：python gui.py

功能：
  - 人脸注册：选择照片文件 + 输入姓名，一键录入特征库
  - 实时签到：摄像头画面内嵌在窗口中，实时显示识别结果
签到逻辑复用 checkin.CheckinEngine，与控制台模式完全一致。
"""
import sys
from pathlib import Path

# 保证从任意目录运行都能找到项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

try:
    from PIL import Image, ImageTk  # noqa: E402
except ImportError:
    print("❌ 未安装 Pillow 库，请先执行：pip install Pillow")
    raise SystemExit(1)

from checkin import CheckinEngine  # noqa: E402
from config import CAMERA_INDEX  # noqa: E402
from register import register_face  # noqa: E402
from utils.drawing import draw_detections  # noqa: E402
from utils.face_detector import FaceDetector  # noqa: E402
from utils.face_embedder import FaceEmbedder  # noqa: E402
from utils.storage import AttendanceRecorder, FeatureDatabase  # noqa: E402

IMG_FILETYPES = [("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]


class RegisterDialog(tk.Toplevel):
    """人脸注册对话框：姓名 + 照片路径"""

    def __init__(self, master, on_success):
        super().__init__(master)
        self.title("人脸注册")
        self.geometry("430x170")
        self.resizable(False, False)
        self.on_success = on_success

        ttk.Label(self, text="学生姓名：").grid(row=0, column=0, padx=10, pady=14, sticky="e")
        self.name_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.name_var, width=24).grid(row=0, column=1, padx=4)

        ttk.Label(self, text="照片路径：").grid(row=1, column=0, padx=10, pady=6, sticky="e")
        self.path_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.path_var, width=24).grid(row=1, column=1, padx=4)
        ttk.Button(self, text="浏览…", command=self._browse).grid(row=1, column=2, padx=4)

        ttk.Button(self, text="开始注册", command=self._register).grid(row=2, column=1, pady=14, sticky="w")

    def _browse(self):
        path = filedialog.askopenfilename(title="选择人脸照片", filetypes=IMG_FILETYPES)
        if path:
            self.path_var.set(path)

    def _register(self):
        name = self.name_var.get().strip()
        path = self.path_var.get().strip()
        if not name or not path:
            messagebox.showwarning("提示", "请填写姓名并选择照片")
            return
        ok, msg = register_face(path, name)
        if ok:
            messagebox.showinfo("注册结果", msg)
            self.on_success()
            self.destroy()
        else:
            messagebox.showerror("注册结果", msg)


class CheckinWindow(tk.Toplevel):
    """实时签到窗口：摄像头画面内嵌展示，tkinter after 定时循环驱动"""

    TICK_MS = 10       # 每 10ms 处理一帧
    MAX_WIDTH = 720    # 画面显示的最大宽度（等比缩放）

    def __init__(self, master):
        super().__init__(master)
        self.title("实时签到")
        self.geometry("780x640")
        self.protocol("WM_DELETE_WINDOW", self._stop)
        self.running = False
        self.cap = None
        self.engine = None

        self.video_label = ttk.Label(self)
        self.video_label.pack(padx=10, pady=6)
        self.info_var = tk.StringVar(value="正在加载检测模型，请稍候…")
        ttk.Label(self, textvariable=self.info_var).pack()
        self.log_text = tk.Text(self, height=7, state="disabled")
        self.log_text.pack(fill="x", padx=10, pady=6)
        ttk.Button(self, text="停止签到", command=self._stop).pack(pady=6)

        self.after(100, self._init)   # 稍作延迟再加载模型，让窗口先显示出来

    def _log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _init(self):
        """初始化检测模型、特征库与摄像头"""
        database = FeatureDatabase().load_all()
        if not database:
            messagebox.showerror("错误", "特征库为空，请先在主界面注册学生人脸")
            self.destroy()
            return
        try:
            detector = FaceDetector()
        except RuntimeError as exc:
            messagebox.showerror("错误", f"人脸检测器初始化失败：{exc}")
            self.destroy()
            return
        embedder = FaceEmbedder()
        try:
            embedder.prepare()   # 下载（如需）并加载特征提取模型
        except RuntimeError as exc:
            messagebox.showerror("错误", str(exc))
            self.destroy()
            return
        engine = CheckinEngine(detector, embedder, database, AttendanceRecorder())
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头，请检查设备或修改 config.py 中的 CAMERA_INDEX")
            self.destroy()
            return
        self.engine, self.cap, self.running = engine, cap, True
        self.info_var.set(f"检测后端：{detector.backend} | 特征库 {len(database)} 人 | 今日已签到 {len(engine.checked_today)} 人")
        self._tick()

    def _tick(self):
        """定时循环：取帧 -> 识别 -> 绘制 -> 更新画面"""
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.info_var.set("读取摄像头画面失败")
            self.after(100, self._tick)
            return
        detections, events = self.engine.process_frame(frame)
        for event in events:
            self._log(event)
        draw_detections(frame, detections)
        self.info_var.set(f"今日已签到 {len(self.engine.checked_today)} 人 | 特征库 {len(self.engine.database)} 人")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if pil.width > self.MAX_WIDTH:
            pil = pil.resize((self.MAX_WIDTH, int(pil.height * self.MAX_WIDTH / pil.width)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.video_label.configure(image=photo)
        self.video_label.image = photo   # 持有引用，防止被垃圾回收
        self.after(self.TICK_MS, self._tick)

    def _stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()


class App(tk.Tk):
    """主窗口：功能菜单"""

    def __init__(self):
        super().__init__()
        self.title("课堂自动人脸签到系统（课程 3320）")
        self.geometry("470x430")
        self.resizable(False, False)

        ttk.Label(self, text="已注册学生", font=("", 13, "bold")).pack(pady=(14, 4))
        self.listbox = tk.Listbox(self, height=11)
        self.listbox.pack(fill="x", padx=24)
        self.count_var = tk.StringVar()
        ttk.Label(self, textvariable=self.count_var).pack(pady=2)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="人脸注册", command=self._open_register).grid(row=0, column=0, padx=6)
        ttk.Button(btn_frame, text="开始签到", command=self._open_checkin).grid(row=0, column=1, padx=6)
        ttk.Button(btn_frame, text="刷新列表", command=self._refresh).grid(row=0, column=2, padx=6)
        ttk.Button(btn_frame, text="退出", command=self.destroy).grid(row=0, column=3, padx=6)

        self._refresh()

    def _refresh(self):
        names = FeatureDatabase().names()
        self.listbox.delete(0, "end")
        for name in names:
            self.listbox.insert("end", name)
        self.count_var.set(f"共 {len(names)} 人 | 签到记录见 data/attendance/ 目录")

    def _open_register(self):
        RegisterDialog(self, self._refresh)

    def _open_checkin(self):
        CheckinWindow(self)


def main():
    try:
        App().mainloop()
    except tk.TclError as exc:
        print("无法启动图形界面（当前环境可能不支持）:", exc)
        print("请使用控制台方式运行：python main.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
