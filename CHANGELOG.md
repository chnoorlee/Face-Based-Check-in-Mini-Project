# 更新日志

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/) 语义化版本规范。

## [v1.2.0] - 2026-11-11 — 商业化发布

**新增（功能完备性）**

- 学生管理：`FeatureDatabase.delete()` / `details()`，支持查看注册时间
- `register.py --list`（列出学生与注册时间）、`--delete 姓名`（删除学生）
- 控制台菜单升级为 5 项：注册 / 签到 / 查看信息（含今日出勤状态）/ 删除学生 / 退出
- `checkin.py --tolerance` 取值范围校验 (0, 2]

**改进（工程质量基线，v1.1.0 内容并入本发布）**

- 配置校验 `validate_config()`：阈值 / 冷却时间 / 置信度 / 摄像头编号合法性检查
- 损坏 CSV 容错、姓名长度限制（≤50 字符）、日志按大小轮转（1MB × 3 份）
- GitHub Actions CI：每次 push / PR 自动运行 52 个单元测试 + 覆盖率门禁（80%，实测 83%）
- 项目元信息 pyproject.toml、pytest-cov 覆盖率配置

**文档**

- GitHub Pages 宣传页（docs/，纯静态零外部依赖）
- 本 CHANGELOG 与 docs/PROMPTS.md（商业化迭代提示词记录）

## [v1.0.0] - 2026-11-11 — 首个完整版本

- 人脸注册模块：静态照片 → YOLO-Face 检测 → SFace 128 维特征 → 本地特征库
- 实时签到模块：摄像头实时识别、冷却期 + 每日去重防重复签到、未知人员标记
- CSV 签到记录（学生姓名 / 签到时间 / 是否签到成功，utf-8-sig 编码）
- 控制台菜单与可选 tkinter 图形界面
- 模型自动下载（多备用地址）与 OpenCV Haar 检测兜底
- 21 个单元测试，端到端实测通过（CPU 约 9.6 FPS）
