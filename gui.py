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
from config import (CAMERA_INDEX, FACES_DIR, PROCESS_EVERY_N_FRAMES,  # noqa: E402
                    PROCESS_MAX_WIDTH)
from register import register_face  # noqa: E402
from utils.drawing import (RosterPanel, attach_roster_panel,  # noqa: E402
                           draw_detections)
from utils.face_detector import FaceDetector  # noqa: E402
from utils.face_embedder import FaceEmbedder  # noqa: E402
from utils.roster import Roster  # noqa: E402
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

    def __init__(self, master, standalone=False):
        super().__init__(master)
        self.standalone = standalone     # True = 一键启动模式（本窗口即主窗口）
        self._closed = False
        if standalone:
            self.title("课堂人脸签到 · 站到摄像头前即可签到（关闭窗口退出）")
            try:
                self.state("zoomed")     # Windows 最大化，方便投影到教室屏幕
            except tk.TclError:
                self.geometry("1100x760")
        else:
            self.title("实时签到")
            self.geometry("780x640")
        self.protocol("WM_DELETE_WINDOW", self._stop)
        self.running = False
        self.cap = None
        self.engine = None
        self._frame_no = 0
        self._last_detections = []

        self.video_label = ttk.Label(self)
        self.video_label.pack(padx=10, pady=6, expand=True)
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

    def _fail(self, msg):
        """初始化失败：提示并关闭窗口（统一出口，避免窗口卡在"加载中"）"""
        messagebox.showerror("错误", msg)
        self._close()

    def _close(self):
        """关闭窗口；一键启动模式下同时结束整个程序"""
        if self._closed:
            return
        self._closed = True
        self.running = False
        try:
            self.destroy()
        finally:
            if self.standalone:
                try:
                    self.master.destroy()
                except tk.TclError:
                    pass

    def _init(self):
        """初始化检测模型、特征库与摄像头（任何异常都要给出可见提示）"""
        cap = None
        try:
            database = FeatureDatabase().load_all()
            if not database:
                self._fail("特征库为空，请先在主界面注册学生人脸")
                return
            detector = FaceDetector()
            embedder = FaceEmbedder()
            embedder.prepare()   # 下载（如需）并加载特征提取模型
            recorder = AttendanceRecorder()
            engine = CheckinEngine(detector, embedder, database, recorder)
            if recorder.last_read_error:
                self._fail(f"今日签到记录读取失败（{recorder.last_read_error}）\n"
                           "请先关闭占用签到 CSV 的程序（如 Excel）后重试")
                return
            try:
                cap = cv2.VideoCapture(CAMERA_INDEX)
            except (cv2.error, TypeError) as exc:
                self._fail(f"摄像头参数不合法（CAMERA_INDEX={CAMERA_INDEX!r}）：{exc}")
                return
            if not cap.isOpened():
                cap.release()
                cap = None
                self._fail("无法打开摄像头，请检查设备或修改 config.py 中的 CAMERA_INDEX")
                return
            # 按时间预热，并检查画面是否可用（挡板关闭/过暗时给出明确提示）
            from utils.camera import diagnose_frame, warmup
            _, probe = warmup(cap)
            if probe is not None:
                healthy, message = diagnose_frame(probe)
                if not healthy:
                    self._log(f"⚠️ 摄像头异常：{message}")
                    messagebox.showwarning("摄像头画面异常", message)
            self.engine, self.cap, self.running = engine, cap, True
            self.roster = Roster.from_database(database)   # 应到名单 = 全部已注册学生
            self.panel = RosterPanel(faces_dir=FACES_DIR)
            self.info_var.set(f"检测后端：{detector.backend} | 应到 {len(self.roster)} 人"
                              f" | 已到 {self.roster.progress_text(engine.checked_today)}")
        except (RuntimeError, cv2.error, OSError) as exc:
            if cap is not None:
                cap.release()      # 失败时务必释放，否则摄像头会被一直占用
            self._fail(f"初始化失败：{exc}")
            return
        self._tick()

    def _tick(self):
        """定时循环：取帧 -> 识别 -> 绘制 -> 更新画面。

        任何异常都不能让循环停摆（否则窗口假死、摄像头不释放），
        统一捕获、记录到日志区，并在 finally 中重新排期。
        """
        if not self.running:
            return
        try:
            self._process_once()
        except (cv2.error, OSError, RuntimeError, ValueError) as exc:
            self._log(f"❌ 运行时错误：{exc}")
            self.info_var.set(f"出错：{exc}")
        finally:
            if self.running:
                self.after(self.TICK_MS, self._tick)

    def _process_once(self):
        """处理并显示一帧"""
        ok, frame = self.cap.read()
        if not ok:
            self.info_var.set("读取摄像头画面失败")
            return
        if PROCESS_MAX_WIDTH and frame.shape[1] > PROCESS_MAX_WIDTH:
            ratio = PROCESS_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PROCESS_MAX_WIDTH, int(frame.shape[0] * ratio)),
                               interpolation=cv2.INTER_AREA)
        # 跳帧：只每 N 帧识别一次，其余帧沿用上次结果，显著降低 CPU 占用
        skip = max(1, int(PROCESS_EVERY_N_FRAMES or 1))
        process_now = self._frame_no % skip == 0      # 第 1 帧（_frame_no=0）必须处理
        self._frame_no += 1
        if process_now:
            detections, events = self.engine.process_frame(frame)
            self._last_detections = detections
        else:
            detections, events = self._last_detections, []
        for event in events:
            self._log(event)
        draw_detections(frame, detections)
        frame = attach_roster_panel(
            frame, self.panel.render(self.roster.names, self.engine.checked_today))
        absent = self.roster.absent(self.engine.checked_today)
        self.info_var.set(f"应到 {len(self.roster)} 人 | 已到 "
                          f"{self.roster.progress_text(self.engine.checked_today)} 人"
                          f" | 缺勤 {len(absent)} 人")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if pil.width > self.MAX_WIDTH:
            pil = pil.resize((self.MAX_WIDTH, int(pil.height * self.MAX_WIDTH / pil.width)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.video_label.configure(image=photo)
        self.video_label.image = photo   # 持有引用，防止被垃圾回收
        # 由 _tick 的 finally 统一重新排期，这里不要再调用 after()

    def _stop(self):
        """关闭窗口：输出出勤汇总并写出勤报告。

        报告写入失败不能影响摄像头释放与窗口关闭，因此清理放在 finally。
        """
        if self._closed:
            return
        self.running = False
        try:
            if getattr(self, "engine", None) is not None and getattr(self, "roster", None):
                try:
                    report = self.engine.recorder.write_summary(
                        self.roster.names, self.engine.checked_today, self.roster.on_leave())
                    present = self.roster.present(self.engine.checked_today)
                    absent = self.roster.absent(self.engine.checked_today)
                    logger_msg = (f"出勤：应到 {len(self.roster)} | 实到 {len(present)} "
                                  f"| 缺勤 {len(absent)} | 未到：{'、'.join(absent) or '无'}")
                    print(logger_msg)          # 无控制台时也写入日志文件
                except OSError as exc:
                    print(f"⚠️ 出勤报告写入失败：{exc}")
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self._close()


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
        ttk.Button(btn_frame, text="退出", command=self._quit).grid(row=0, column=3, padx=6)

        self._checkin = None
        self._refresh()

    def _refresh(self):
        try:
            names = FeatureDatabase().names()
        except OSError as exc:
            messagebox.showerror("错误", f"读取特征库失败：{exc}")
            return
        self.listbox.delete(0, "end")
        for name in names:
            self.listbox.insert("end", name)
        self.count_var.set(f"共 {len(names)} 人 | 签到记录见 data/attendance/ 目录")

    def _open_register(self):
        RegisterDialog(self, self._refresh)

    def _open_checkin(self):
        self._checkin = CheckinWindow(self)

    def _quit(self):
        """退出前先关闭签到窗口，确保出勤报告已写出、摄像头已释放"""
        win = getattr(self, "_checkin", None)
        try:
            if win is not None and win.winfo_exists():
                win._stop()
        except Exception:
            pass
        self.destroy()


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
