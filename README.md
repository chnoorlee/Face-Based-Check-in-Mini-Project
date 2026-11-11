# 课堂自动人脸签到系统（课程 3320）

[![CI](https://github.com/chnoorlee/Face-Based-Check-in-Mini-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/chnoorlee/Face-Based-Check-in-Mini-Project/actions)
[![Release](https://img.shields.io/badge/release-v1.2.0-7c6cff)](https://github.com/chnoorlee/Face-Based-Check-in-Mini-Project/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 🌐 项目宣传页（GitHub Pages）：https://chnoorlee.github.io/Face-Based-Check-in-Mini-Project/

基于 **Python + YOLO-Face + 人脸特征比对** 的课堂自动签到课程项目。

- **人脸注册模块**：上传静态照片 → YOLO-Face 检测人脸 → 提取 128 维人脸特征 → 保存到本地特征库
- **多照片登记**：同一学生可登记 2~5 张照片（戴/不戴眼镜、不同光线），识别时取最接近的一张——这是提升真实场景识别率最有效的办法
- **实时签到模块**：调用本地摄像头实时画面 → YOLO-Face 检测画面中的人脸 → 与特征库比对识别 → 自动完成签到
- **自动打勾**：画面右侧名单侧栏，识别到本人立刻打 ✅，实时显示「已到 X / 应到 Y」
- **四态考勤**：已到 / **迟到**（超过上课时间 + 宽限）/ **请假** / 缺勤
- **缺勤名单**：签到结束时自动输出**没来的人**，并生成出勤报告 CSV
- **节次支持**：同一天多节课用 `--session 第1节`，签到去重按节次独立，互不影响
- **出勤统计**：`stats.py` 汇总历史课次，输出每人出勤率表（含迟到/请假次数）
- **环境自检**：`doctor.py` 一键检查依赖、模型、摄像头、字体、目录权限
- **防重复签到**：冷却期 + 每日去重，同一学生短时间内不会重复签到
- **签到记录**：本地 CSV 表格（学生姓名 / 签到时间 / 是否签到成功），Excel 可直接打开
- **未知人员**：未匹配特征库的人脸在画面中标记为「未知人员」，不写入签到记录

> **应到名单怎么来？** 默认 = 特征库中所有已注册学生；也可用 `--roster 名单.txt` 指定本次课堂的名单（每行一个姓名，或 Excel 导出的 CSV 第一列）。名单里有人还没注册人脸时，程序启动会明确提示。

---

## 1. 项目结构

```
face/
├── config.py                  # 全局配置（阈值、冷却时间、摄像头编号等）
├── register.py                # 人脸注册模块（静态图片）
├── checkin.py                 # 实时签到模块（摄像头 / 视频流）
├── main.py                    # 控制台主菜单入口
├── stats.py                   # 出勤统计报表（历史出勤率 CSV）
├── doctor.py                  # 环境自检（依赖/模型/摄像头/字体）
├── gui.py                     # 可选 tkinter 图形界面
├── requirements.txt           # 依赖清单
├── pyproject.toml             # 项目元信息与 pytest 配置
├── utils/
│   ├── face_detector.py       # YOLO-Face 人脸检测（模型缺失时自动回退 Haar）
│   ├── face_embedder.py       # 人脸特征提取（SFace + YuNet 对齐，OpenCV DNN）
│   ├── storage.py             # 特征库 + 签到 CSV（含节次）+ 四态出勤报告 + 特征比对
│   ├── roster.py              # 应到名单（文件/特征库）、请假登记与出勤统计
│   ├── drawing.py             # 画面标注（识别框 + 姓名标签 + 名单打勾侧栏）
│   └── imutils.py             # 中文路径图片读写、模型下载等小工具
├── scripts/
│   ├── make_demo_video.py     # 演示视频生成（无摄像头环境演示用）
│   └── make_og_image.py       # 宣传页社交分享图生成
├── docs/                      # GitHub Pages 宣传页（纯静态，零外部依赖）
├── tests/                     # 单元测试（存储 / 比对 / 防重复签到逻辑）
├── data/                      # 本地数据（运行后生成，不入库）
│   ├── faces/                 #   注册的人脸照片
│   ├── embeddings/            #   人脸特征向量库（每人一个 .npy）
│   ├── attendance/            #   签到 CSV（每天一个文件）
│   └── logs/                  #   运行日志
└── models/                    # yolov8n-face.pt（首次运行自动下载）
```

## 2. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows / macOS / Linux |
| Python | 3.9 及以上（推荐 3.10 ~ 3.12） |
| 摄像头 | 签到阶段需要（无摄像头可用视频文件演示，见第 7 节） |
| 网络 | 首次运行需自动下载 3 个模型文件（共约 43MB） |

> 检查 Python 版本：`python --version`

## 3. 安装依赖

在项目目录下执行（Windows 打开 CMD/PowerShell，进入 face 文件夹）：

```bash
# 1. 创建虚拟环境（推荐）
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# macOS / Linux 激活
source .venv/bin/activate

# 2. （推荐）先安装 CPU 版 PyTorch，体积比默认版小很多，YOLO 推理用 CPU 完全够
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其余依赖
pip install -r requirements.txt
```

**国内网络加速**：为第 2、3 步加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
（第 2 步的 `--index-url` 是 PyTorch 官方 CPU 源，若下载慢可改用 `pip install torch torchvision` 走镜像，只是体积较大）。

**依赖说明**：

| 库 | 作用 |
|----|------|
| ultralytics + torch | YOLO 模型推理框架 |
| opencv-python | 摄像头采集、画面处理，以及人脸特征提取（内置 DNN 模块跑 SFace 模型） |
| Pillow | 画面中文标签绘制、GUI 图像显示 |

> 为什么不使用常见的 face_recognition（dlib）库？dlib 在 Windows 上没有官方预编译包，
> 需要安装 CMake + Visual Studio 编译工具链才能从源码编译，对课程作业不够"开箱即用"。
> 本项目改用 OpenCV 自带的 DNN 模块运行 SFace 模型，纯 pip 安装即可使用，效果同样可靠。

**模型文件**（首次运行自动下载到 models/ 目录，各 2~3 个备用地址）：

| 模型 | 大小 | 用途 |
|------|------|------|
| yolov8n-face.pt | 约 6MB | YOLO-Face 人脸检测 |
| face_recognition_sface_2021dec.onnx | 约 37MB | 人脸特征提取（128 维） |
| face_detection_yunet_2023mar.onnx | 约 230KB | 人脸关键点对齐 |

## 4. 快速开始

### 方式一：控制台菜单（推荐）

```bash
python main.py
```

按菜单选择：`1` 人脸注册 → `2` 实时签到 → `3` 查看信息（注册时间 + 今日出勤）→ `4` 删除学生 → `5` 退出。

### 方式二：直接运行各模块

**第一步：人脸注册（静态图片）**

```bash
python register.py -i 自己的照片.jpg -n 张三
# 强烈建议一次登记 2~3 张不同状态的照片（戴/不戴眼镜、不同光线）：
python register.py -i 正面.jpg -i 戴眼镜.jpg -i 侧光.jpg -n 张三
# 或交互式：python register.py
```

> 同名重复注册默认**追加**一张照片（最多 5 张）；要清空重来加 `--replace`。
> `python register.py --list` 可查看每人已登记几张照片。

输出示例（首次运行会自动下载模型）：

```
[模型下载] 未找到 yolov8n-face.pt，正在自动下载（约 6MB）...
[模型下载] YOLO-Face（约 6MB） 下载完成：models\yolov8n-face.pt（6.0MB）
[模型下载] SFace（约 37MB） 下载完成：models\face_recognition_sface_2021dec.onnx（36.9MB）
✅ 注册成功！姓名：张三 | 检测后端：YOLO-Face
```

注册成功后：特征向量保存到 `data/embeddings/张三.npy`，照片备份到 `data/faces/张三.jpg`。
可继续注册多名同学；**同名注册会覆盖旧记录**。

**第二步：实时签到（摄像头动态识别）**

```bash
python checkin.py                                    # 应到名单 = 全部已注册学生
python checkin.py --roster 名单.txt                   # 指定本次课堂应到名单
python checkin.py -S 第1节 --leave 王五 --start 08:00  # 节次 + 请假 + 上课时间（判迟到）
```

画面中会实时出现识别结果，右侧是**签到名单侧栏**（谁到了就打勾）：

- 绿色框「张三 · 签到成功」→ 自动写入 CSV，侧栏对应名字打 ✅
- 黄色框「张三 · 冷却中 Ns / 已签到」→ 不重复签到
- 红色框「未知人员」→ 不在特征库中，只标记不签到

按 `q` 退出，控制台输出出勤汇总 + **缺勤名单**：

```
==============================================
出勤：应到 3 人 | 实到 1 人 | 请假 1 人 | 缺勤 1 人
✅ 已到（1），其中迟到 1 人：张三
⏰ 迟到（1）：张三
📝 请假（1）：王五
❌ 未到（1）：李四
签到明细：data\attendance\attendance_2026-11-11_第1节.csv
出勤报告：data\attendance\attendance_2026-11-11_第1节_report.csv
```

> **一天多节课**：用 `-S 第1节` / `-S 第2节` 区分节次，记录分文件存放，去重互不影响——同一学生上午签过，下午仍能正常签到。
>
> **迟到判定**：签到时间晚于「上课时间 + 迟到宽限」（默认 08:00 + 10 分钟）记为迟到，可在 `config.py` 或 `--start` 调整。

**第三步：查看签到记录**

打开 `data/attendance/attendance_2026-11-11.csv`（Excel / 记事本均可）：

| 学生姓名 | 签到时间 | 是否签到成功 |
|----------|---------------------|--------------|
| 张三 | 2026-11-11 10:23:45 | 是 |

再打开同目录的 `attendance_2026-11-11_report.csv`，是**按应到名单逐人标记**的出勤报告：

| 学生姓名 | 状态 | 签到时间 |
|----------|------|---------------------|
| 张三 | 已到 | 2026-11-11 10:23:45 |
| 李四 | 缺勤 | |

> 不接摄像头也能补出缺勤名单：`python main.py` 菜单选 `4. 出勤报告`，或直接运行 `python register.py --list` 查看注册情况。

### 方式三：tkinter 图形界面（可选）

```bash
python gui.py
```

主界面可注册学生、查看特征库；「开始签到」后摄像头画面内嵌在窗口中实时显示识别结果。

## 5. 命令行参数

| 命令 | 说明 |
|------|------|
| `python register.py -i 照片.jpg -n 张三` | 注册学生（-i 照片路径，-n 姓名；`-i` 可重复登记多张） |
| `python register.py -i 新照片.jpg -n 张三 --replace` | 覆盖重建该学生的登记 |
| `python register.py --list` | 列出已注册学生与注册时间 |
| `python register.py --delete 张三` | 删除学生 |
| `python checkin.py` | 摄像头实时签到（应到名单 = 全部已注册学生） |
| `python checkin.py --roster 名单.txt` | 指定本次课堂应到名单（.txt 每行一个 / .csv 第一列） |
| `python checkin.py -S 第1节` | 指定节次（同日多节课互不影响） |
| `python checkin.py -L 王五 -L 赵六` | 登记请假（可重复） |
| `python checkin.py --start 13:30` | 指定上课时间，用于判定迟到 |
| `python checkin.py -t 1.2` | 指定识别阈值（临时覆盖 config，取值 (0, 2]） |
| `python stats.py --roster 名单.txt` | 汇总历史出勤率并导出 `出勤统计.csv` |
| `python doctor.py` | 环境自检（依赖 / 模型 / 摄像头 / 字体 / 目录） |
| `python checkin.py --source 视频.mp4 --no-window` | 用视频文件演示签到，不显示画面窗口 |
| `python main.py` | 控制台菜单 |
| `python gui.py` | 图形界面 |
| `python -m pytest tests/ -v` | 运行单元测试（需 pip install pytest pytest-cov） |

## 6. 核心原理

```
注册阶段（静态图片）                签到阶段（摄像头实时画面）
     │                                    │
  上传照片                            摄像头取帧
     │                                    │
YOLO-Face 人脸检测                   YOLO-Face 人脸检测
     │                                    │
YuNet 关键点对齐人脸                YuNet 关键点对齐人脸
     │                                    │
SFace 提取 128 维特征               SFace 提取 128 维特征
     │                                    │
保存到特征库 .npy ──────────────► 与特征库逐一比对（欧氏距离）
                                          │
                                    距离 ≤ 阈值 → 识别成功 → 签到（写入 CSV）
                                          │              → 名单侧栏打 ✅
                                          │
                                    距离 > 阈值 → 未知人员（仅画面标记）
                                          │
                                    退出时 → 对照应到名单 → 输出缺勤名单 + 出勤报告
```

1. **YOLO-Face 检测**：`yolov8n-face.pt` 是 YOLOv8 的轻量人脸检测模型，模型文件缺失时程序会自动下载（多个备用地址依次尝试），全部失败则回退到 OpenCV 自带 Haar 检测器，保证程序始终可运行；
2. **关键点对齐**：YuNet 在人脸区域上提取 5 个关键点（双眼/鼻尖/嘴角），通过相似变换把人脸对齐到标准位置，减少角度、位置差异对识别的影响；
3. **特征提取与比对**：SFace 模型（OpenCV DNN 推理）把对齐后的人脸编码为 128 维归一化向量，同一人不同照片的向量距离接近、不同人距离远，用**欧氏距离** + 阈值判断是否同一人；
4. **防重复签到**：签到成功后进入 `COOLDOWN_SECONDS` 秒冷却期（画面标记「冷却中 N 秒」）；同时当天只签到一次，程序重启后读取当天 CSV 继续生效；
5. **记录落盘**：签到事件即时追加写入当天 CSV（utf-8-sig 编码，Excel 打开中文不乱码）；
6. **出勤闭环**：应到名单（默认 = 特征库全部学生，或 `--roster` 指定）→ 识别到人立刻打 ✅ → 退出时对照名单算出**缺勤名单**并写出勤报告；名单里未注册人脸的学生会被提前提示「无法识别」。

## 7. 无摄像头环境演示

没有摄像头也可以完整演示签到流程：

```bash
# 1. 注册
python register.py -i 照片.jpg -n 张三

# 2. 把照片生成演示视频（60 帧）
python scripts/make_demo_video.py 照片.jpg demo.mp4

# 3. 用视频流演示签到
python checkin.py --source demo.mp4 --no-window
```

## 8. 可调参数（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MATCH_TOLERANCE` | 1.0 | 识别阈值（欧氏距离，特征已归一化取值 0~2）。认不出自己 → 调大到 1.2；误识别 → 调小到 0.8 |
| `CLASS_START_TIME` | `"08:00"` | 上课时间，用于判定迟到（也可用 `--start` 临时覆盖） |
| `LATE_GRACE_MINUTES` | 10 | 迟到宽限（分钟）：08:10 之后签到记为迟到 |
| `COOLDOWN_SECONDS` | 60 | 同一学生重复签到的冷却时间（秒） |
| `DETECT_CONFIDENCE` | 0.5 | YOLO 检测置信度阈值 |
| `CAMERA_INDEX` | 0 | 摄像头编号（外接摄像头可改为 1） |
| `SHOW_WINDOW` | True | 是否显示实时画面窗口 |

## 9. 常见问题（FAQ）

**Q1：pip 安装依赖报错？**
→ 确认 Python 是 64 位（`python -c "import struct;print(struct.calcsize('P')*8)"` 输出 64）；
国内网络加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；
torch 下载慢时改用 CPU 版（见第 3 节第 2 步）；
若已安装 OpenCV 5 导致模型加载失败，降级：`pip install "opencv-python<5"`（requirements.txt 已锁定）。

**Q2：运行签到提示「无法打开摄像头」？**
→ 检查摄像头是否被腾讯会议等程序占用；在 config.py 中把 `CAMERA_INDEX` 改为 1 试试；或用第 7 节的视频方式演示。

**Q3：模型自动下载失败？**
→ 手动下载对应文件放到 `models/` 目录（完整地址列表见 config.py）：
[yolov8n-face.pt](https://github.com/derronqi/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt)、
[SFace](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)、
[YuNet](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)。
注意 SFace/YuNet 在 opencv_zoo 仓库中是 Git LFS 存储，需用上面的 media 地址（raw 地址只会下载到 130 字节的指针文件）。
YOLO 模型缺失时检测自动回退 Haar（精度略低），特征模型缺失则需下载后才能注册/签到。

**Q4：控制台输出中文乱码 / 报 UnicodeEncodeError？**
→ 程序已自动把输出编码改为 UTF-8；若仍有问题，Windows 可在终端执行 `chcp 65001`。

**Q5：识别不准确 / 换了样子就认不出？**
→ 先**多登记几张照片**（这是最有效的一招）。实测同一人「戴眼镜」与「不戴眼镜」两张照片的
特征距离可达 1.33，与「不同的人」几乎一样远——模型分不开，但把两张都登记进去，
签到时会取最接近的一张，问题即解决：

```bash
python register.py -i 戴眼镜.jpg -i 不戴眼镜.jpg -n 张三
```

其次：注册照片要光线充足、正面无遮挡；正对摄像头、避免侧脸和逆光；必要时按第 8 节调整 `MATCH_TOLERANCE`。

**Q6：照片路径含中文读不出来？**
→ 程序已内置中文路径支持（`utils/imutils.py` 的 `imread_unicode`），可直接使用。

**Q7：为什么未知人员不写入 CSV？**
→ 签到表只记录「成功签到」事件（是否签到成功=是）。未知人员仅在画面中标记红色框并在控制台提示，避免签到表被无效记录刷屏。

## 10. 技术栈与许可

| 组件 | 用途 | 许可 |
|------|------|------|
| YOLOv8-Face（ultralytics） | 人脸检测 | GPL-3.0 |
| SFace / YuNet（OpenCV Zoo） | 特征提取与关键点对齐 | Apache-2.0 |
| OpenCV | 视频采集、图像处理、DNN 推理 | Apache-2.0 |

本项目仅用于课程教学演示，请遵守各组件许可协议。

## 11. 课程作业提示

- 演示流程建议：先注册 2~3 名同学 → 运行签到，让同学依次走过摄像头 → 展示实时识别画面与 CSV 记录 → 演示冷却期（同一人来回走不重复签到）与未知人员标记
- 代码注释完整，模块划分清晰（注册/签到/存储/检测各司其职），可按模块讲解
- `tests/` 目录包含单元测试，可运行 `python -m pytest tests/ -v` 展示测试通过结果
