# -*- coding: utf-8 -*-
"""
实时摄像头签到模块

流程：摄像头取帧 -> YOLO-Face 人脸检测 -> 提取人脸特征 -> 与特征库比对 -> 自动签到

防重复签到设计：
  1. 冷却期：学生签到成功后 COOLDOWN_SECONDS 秒内不再重复签到（画面标记"冷却中"）；
  2. 每日去重：同一学生当天只签到一次，程序重启后读取当天 CSV 继续生效；
  3. 未知人员：与特征库都不匹配的人脸标记为"未知人员"，不写入签到记录。

签到结果写入 data/attendance/attendance_YYYY-MM-DD.csv（学生姓名/签到时间/是否签到成功）。

用法：
    python checkin.py                                # 摄像头实时签到
    python checkin.py --source 视频.mp4 --no-window  # 无摄像头环境用视频演示
"""
import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 保证从任意目录运行都能找到项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Windows 控制台默认 GBK 编码，改为 UTF-8 避免中文/表情符号输出报错
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv2  # noqa: E402

from config import (CAMERA_INDEX, COOLDOWN_SECONDS, FACES_DIR,  # noqa: E402
                    MATCH_TOLERANCE, PROCESS_EVERY_N_FRAMES,
                    PROCESS_MAX_WIDTH, SHOW_WINDOW, WINDOW_NAME,
                    late_after as config_late_after, validate_config)
from utils.drawing import (RosterPanel, attach_roster_panel,  # noqa: E402
                           draw_detections)
from utils.face_detector import FaceDetector  # noqa: E402
from utils.face_embedder import FaceEmbedder  # noqa: E402
from utils.roster import Roster  # noqa: E402
from utils.storage import (AttendanceRecorder, FeatureDatabase,  # noqa: E402
                           find_best_match, is_late)

logger = logging.getLogger(__name__)

# 状态 -> 画面标签文字（与 utils/drawing.py 中的 _STATUS_TEXT 保持一致）
STATUS_TEXT = {
    "new": "签到成功",
    "cooldown": "冷却中",
    "checked_in": "已签到",
    "unknown": "未知人员",
    "error": "写入失败",
}


@dataclass
class Detection:
    """一帧画面中一个人脸的识别结果"""
    box: tuple                          # (x1, y1, x2, y2)
    name: Optional[str] = None          # 识别出的姓名；未知人员为 None
    distance: Optional[float] = None    # 与特征库最佳匹配的距离
    status: str = "unknown"             # new / cooldown / checked_in / unknown
    cooldown_left: Optional[float] = None  # 冷却剩余秒数（status=cooldown 时有值）


class CheckinEngine:
    """签到核心引擎：识别比对 + 防重复签到（冷却期 + 每日去重）。

    引擎只负责逻辑（不负责画面显示与设备读写），
    控制台程序与 tkinter 图形界面共用同一个引擎。
    """

    UNKNOWN_LOG_INTERVAL = 3.0   # 未知人员控制台提示的最小间隔（秒），避免刷屏

    def __init__(self, detector, embedder, database, recorder,
                 tolerance=MATCH_TOLERANCE, cooldown=COOLDOWN_SECONDS):
        self.detector = detector
        self.embedder = embedder
        self.database = database                      # {姓名: 特征向量}
        self.recorder = recorder
        self.tolerance = tolerance
        self.cooldown = cooldown
        self.checked_today = recorder.load_checked_names()  # 今日已签到名单（含 CSV 历史）
        self.last_checkin = {}                        # 姓名 -> 最近签到时刻（冷却判断用）
        self.new_today = []                           # 本次运行新签到的名单（用于汇总）
        self._last_unknown_log = 0.0
        self._write_errors = set()                    # 已提示过的写入失败学生（避免刷屏）

    def process_frame(self, frame):
        """处理一帧画面。返回 (识别结果列表, 事件消息列表)。"""
        detections, events = [], []
        for box in self.detector.detect(frame):
            embedding = self.embedder.extract(frame, box)
            if embedding is None:
                continue    # 人脸太小或特征提取失败，跳过
            name, dist = find_best_match(embedding, self.database, self.tolerance)
            cooldown_left = None
            if name is None:
                # 与特征库都不匹配 -> 未知人员，只标记不签到
                status = "unknown"
                now = time.monotonic()
                if now - self._last_unknown_log >= self.UNKNOWN_LOG_INTERVAL:
                    self._last_unknown_log = now
                    events.append("⚠️ 检测到未知人员（不在特征库中）")
            elif name in self.checked_today:
                # 今天已签过：冷却期内显示"冷却中（剩余秒数）"，之后显示"已签到"
                if self._in_cooldown(name):
                    status = "cooldown"
                    cooldown_left = self.cooldown - (time.monotonic() - self.last_checkin[name])
                else:
                    status = "checked_in"
            else:
                # 第一次识别到该学生 -> 自动完成签到
                try:
                    row = self.recorder.record(name)
                except OSError as exc:
                    # 磁盘满 / 文件被 Excel 占用 / 无写权限：提示但不中断整场签到
                    status = "error"
                    if name not in self._write_errors:
                        self._write_errors.add(name)
                        events.append(f"❌ {name} 签到写入失败：{exc}")
                        events.append("   请检查签到 CSV 是否被 Excel 打开、磁盘是否已满")
                else:
                    self.last_checkin[name] = time.monotonic()
                    self.checked_today.add(name)
                    self.new_today.append(name)
                    status = "new"
                    events.append(f"✅ [签到成功] {name}（{row['签到时间']}）")
            detections.append(Detection(box=box, name=name, distance=dist,
                                        status=status, cooldown_left=cooldown_left))
        return detections, events

    def _in_cooldown(self, name) -> bool:
        """该学生当前是否处于冷却期内"""
        last = self.last_checkin.get(name)
        return last is not None and time.monotonic() - last < self.cooldown


def run_console(source=None, show_window=SHOW_WINDOW, tolerance=MATCH_TOLERANCE,
                roster=None, session=None,
                late_after=None):  # pragma: no cover - 摄像头/视频硬件循环
    """控制台方式运行实时签到。

    source：摄像头编号（int）或视频文件路径（str），None 表示使用默认摄像头。
    tolerance：识别阈值，可临时覆盖 config.py 中的 MATCH_TOLERANCE（便于课堂演示调参）。
    roster：应到名单（Roster 对象）。为 None 时取特征库中全部已注册学生。
    session：节次名称（如「第1节」）。同一天多节课时用不同节次，去重互相独立。
    late_after：迟到判定时刻（datetime.time）；None 时按 config 中的上课时间 + 宽限计算。
    返回退出码：0 成功，1 失败。
    """
    database = FeatureDatabase().load_all()
    if not database:
        print("❌ 特征库为空。请先注册学生人脸：python register.py -i 照片.jpg -n 姓名")
        return 1
    print(f"已加载特征库：{len(database)} 人（{'、'.join(database)}）")

    # 应到名单：默认全部已注册学生；指定名单文件时以文件为准
    roster = roster if roster is not None else Roster.from_database(database)
    if not len(roster):
        print("❌ 应到名单为空，无法统计出勤")
        return 1
    missing = roster.unregistered(database)
    if missing:
        print(f"⚠️ 名单中 {len(missing)} 人未注册人脸，无法被识别：{'、'.join(missing)}")
    print(f"应到 {len(roster)} 人：{'、'.join(roster.names)}")

    print("正在初始化人脸检测模型（首次运行需要一点时间）...")
    try:
        detector = FaceDetector()
    except RuntimeError as exc:
        print("❌ 人脸检测器初始化失败：", exc)
        return 1
    print(f"人脸检测后端：{detector.backend}")

    embedder = FaceEmbedder()
    try:
        embedder.prepare()   # 下载（如需）并加载特征提取模型
    except RuntimeError as exc:
        print("❌", exc)
        return 1

    recorder = AttendanceRecorder(session=session)
    if session:
        print(f"本节次：{session}（签到记录：{recorder.path.name}）")
    if late_after is None:
        late_after = config_late_after()
    engine = CheckinEngine(detector, embedder, database, recorder, tolerance=tolerance)
    if recorder.last_read_error:
        print(f"⚠️ 今日签到记录读取失败（{recorder.last_read_error}）")
        print("   为免重复签到，请先关闭占用该文件的程序（如 Excel）后重新运行")
        return 1
    print(f"今日已签到：{len(engine.checked_today)} 人；应到：{len(roster)} 人"
          f"（已到 {roster.progress_text(engine.checked_today)}）")
    if roster.on_leave():
        print(f"请假：{'、'.join(roster.on_leave())}")
    print(f"冷却时间：{engine.cooldown} 秒；识别阈值：{engine.tolerance}"
          + (f"；迟到线：{late_after:%H:%M}" if late_after else ""))

    camera_source = source if source is not None else CAMERA_INDEX
    try:
        cap = cv2.VideoCapture(camera_source)
    except (cv2.error, TypeError) as exc:
        # 例如 config.CAMERA_INDEX 被写成 0.5 之类的非整数
        print(f"❌ 摄像头参数不合法（{camera_source!r}）：{exc}")
        print("   请在 config.py 中把 CAMERA_INDEX 设为整数（0、1……）")
        return 1
    if not cap.isOpened():
        print("❌ 无法打开摄像头。请检查：")
        print("   1) 摄像头是否被其他程序（如腾讯会议）占用")
        print("   2) 系统摄像头权限是否开启")
        print("   3) 可在 config.py 中修改 CAMERA_INDEX（外接摄像头可尝试 1）")
        print("   演示可用视频文件：python checkin.py --source 视频.mp4 --no-window")
        return 1

    if not isinstance(source, str):
        # 实测：首帧几乎全黑，自动曝光约需 1~2 秒才稳定，因此按时间预热而不是按帧数
        print("摄像头预热中（等待自动曝光稳定，约 1.5 秒）...")
        from utils.camera import diagnose_frame, warmup
        count, probe = warmup(cap)
        print(f"  已预热 {count} 帧")
        if probe is not None:
            healthy, message = diagnose_frame(probe)
            print(("✅ 摄像头" if healthy else "⚠️ 摄像头异常：") + message)
            if not healthy:
                print("   （仍会继续运行，但在此画面下无法识别人脸）")

    print("开始实时签到...（按 q 退出）")
    frames, t0, show = 0, time.monotonic(), show_window
    panel = RosterPanel(faces_dir=FACES_DIR)
    ever_visible = False   # 窗口是否曾经可见（用于区分"窗口被用户关闭"与"环境本就不显示窗口"）
    skip = max(1, int(PROCESS_EVERY_N_FRAMES or 1))   # 每 N 帧识别一次
    last_detections = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("视频播放结束。" if isinstance(source, str) else "❌ 读取摄像头画面失败")
                break

            # 高分辨率画面先缩小再推理（实测 1080p -> 960 宽 提速约 4 倍）
            if PROCESS_MAX_WIDTH and frame.shape[1] > PROCESS_MAX_WIDTH:
                ratio = PROCESS_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (PROCESS_MAX_WIDTH, int(frame.shape[0] * ratio)),
                                   interpolation=cv2.INTER_AREA)

            # 跳帧：只在第 N 帧做识别，其余帧沿用上次结果（签到不要求 30 帧连续推理）
            if frames % skip == 0:
                detections, events = engine.process_frame(frame)
                last_detections = detections
            else:
                detections, events = last_detections, []
            new_checkin = False
            for event in events:
                print(event)
                if event.startswith("✅"):
                    new_checkin = True
            if new_checkin:
                print(f"   → 已到 {roster.progress_text(engine.checked_today)} 人")
            frames += 1

            if show:
                draw_detections(frame, detections)   # 仅显示画面时才绘制标注（无窗口模式省去 PIL 开销）
                frame = attach_roster_panel(
                    frame, panel.render(roster.names, engine.checked_today))
                fps = frames / max(time.monotonic() - t0, 1e-6)
                cv2.putText(frame, f"FPS {fps:.1f} | {roster.progress_text(engine.checked_today)} | Press q to quit",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                try:
                    cv2.imshow(WINDOW_NAME, frame)
                    key = cv2.waitKey(1) & 0xFF
                    # 窗口可见性：无桌面会话等环境下窗口永远不可见（返回 0），不能当作
                    # "用户关闭窗口"；仅在窗口曾可见、之后变为不可见时才退出
                    visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                    if visible >= 1:
                        ever_visible = True
                    if key == ord("q") or (ever_visible and 0 <= visible < 1):
                        break
                except cv2.error:
                    # 无图形环境（如服务器），自动切换为纯控制台模式
                    show = False
                    print("当前环境不支持图形窗口，已切换为纯控制台模式（Ctrl+C 结束）")
            else:
                time.sleep(0.01)   # 无窗口模式稍作休眠，避免占满 CPU
    except KeyboardInterrupt:
        print("\n收到中断，正在退出...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - t0
    records = recorder.load_records()
    present = roster.present(engine.checked_today)
    on_leave = roster.on_leave()
    absent = roster.absent(engine.checked_today)
    late_names = [n for n in present if is_late(records.get(n, ""), late_after)]
    report_path = recorder.write_summary(
        roster.names, engine.checked_today, on_leave, late_after)

    print("=" * 46)
    print(f"本次运行新签到 {len(engine.new_today)} 人：{'、'.join(engine.new_today) or '无'}")
    print(f"出勤：应到 {len(roster)} 人 | 实到 {len(present)} 人 | "
          f"请假 {len(on_leave)} 人 | 缺勤 {len(absent)} 人")
    print(f"✅ 已到（{len(present)}）{'，其中迟到 ' + str(len(late_names)) + ' 人' if late_names else ''}："
          f"{'、'.join(present) or '无'}")
    if late_names:
        print(f"⏰ 迟到（{len(late_names)}）：{'、'.join(late_names)}")
    if on_leave:
        print(f"📝 请假（{len(on_leave)}）：{'、'.join(on_leave)}")
    print(f"❌ 未到（{len(absent)}）：{'、'.join(absent) or '无'}")
    print(f"签到明细：{recorder.path}")
    if report_path is None:
        print("⚠️ 签到记录读取失败，本次未生成出勤报告（原报告保持不变）")
    else:
        print(f"出勤报告：{report_path}")
    print(f"平均处理速度：{frames / max(elapsed, 1e-6):.1f} 帧/秒")
    return 0


def main():  # pragma: no cover - 命令行入口（手工实测覆盖）
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for warning in validate_config():
        print("⚠️ 配置警告：", warning)
    parser = argparse.ArgumentParser(description="课堂人脸签到系统 - 实时签到模块")
    parser.add_argument("--source", "-s", help="视频源：摄像头编号或视频文件路径（默认摄像头 0）")
    parser.add_argument("--no-window", action="store_true", help="不显示画面窗口，仅控制台输出")
    parser.add_argument("--tolerance", "-t", type=float, default=MATCH_TOLERANCE,
                        help=f"识别阈值（欧氏距离，默认 {MATCH_TOLERANCE}，越小越严格）")
    parser.add_argument("--roster", "-r", metavar="名单文件",
                        help="应到名单（.txt 每行一个姓名 / .csv 第一列）；不指定则默认全部已注册学生")
    parser.add_argument("--session", "-S", metavar="节次",
                        help="节次名称，如「第1节」。同一天多节课用不同节次，签到去重互不影响")
    parser.add_argument("--leave", "-L", action="append", default=[], metavar="姓名",
                        help="登记请假（可重复使用：--leave 张三 --leave 李四）")
    parser.add_argument("--start", metavar="HH:MM",
                        help="本节课上课时间，用于判定迟到（默认取 config.CLASS_START_TIME）")
    args = parser.parse_args()

    if not 0 < args.tolerance <= 2:
        print("❌ --tolerance 必须在 (0, 2] 范围内（归一化特征欧氏距离取值 0~2）")
        return 1

    late_after = config_late_after(args.start)
    if args.start and late_after is None:
        print("❌ --start 格式应为 HH:MM，例如 08:00")
        return 1

    roster = None
    if args.roster:
        try:
            roster = Roster.from_file(args.roster, leave=args.leave)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print("❌ 读取名单失败：", exc)
            return 1
        if not len(roster):
            print("❌ 名单文件为空：", args.roster)
            return 1
        print(f"已加载应到名单：{args.roster}（{len(roster)} 人）")
    elif args.leave:
        # 未指定名单文件：请假名单在加载特征库后再收敛（只保留名单内的学生）
        pass

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    if roster is None and args.leave:
        from utils.storage import FeatureDatabase as _DB
        roster = Roster.from_database(_DB().load_all(), leave=args.leave)

    if args.leave and roster is not None:
        ignored = [n for n in args.leave if n not in roster.leave]
        if ignored:
            print(f"⚠️ 以下请假人员不在应到名单中，已忽略：{'、'.join(ignored)}")

    return run_console(source=source, show_window=not args.no_window,
                       tolerance=args.tolerance, roster=roster,
                       session=args.session, late_after=late_after)


if __name__ == "__main__":
    raise SystemExit(main())
