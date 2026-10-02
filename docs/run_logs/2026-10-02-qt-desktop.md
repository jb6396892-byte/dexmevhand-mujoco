# 2026-10-02 Qt 抓杯界面与 OpenGL 修复

## 任务与保护范围

新增桌面指令输入、实时仿真与关键参数；修复 `Missing GL version`。不修改阶段 3 策略、示范、物理参数，不重新训练，不修改阶段 6 `study_v4_guard2` 冻结源文件和 checkpoint。新增 Qt 依赖与执行记录只在共享盘 `lora/language/` 下；代码和少量答辩证据留在项目。

## 实际排查顺序

1. 原生最小窗口重现 CPU 扩展下的 GLEW `Missing GL version`，确认不是模型推理失败。
2. 选择已有隔离 GPU 扩展，原生窗口 probe 成功，实际渲染器为 RTX 4060 Laptop / OpenGL 4.6.0。
3. 离屏隐藏窗口遇到单缓冲 GLXFBConfig 不支持，采用仅在物理子进程生效的隐藏双缓冲窗口工厂，MuJoCo FBO 成功连续输出图像。
4. 为 Qt 建立独立包目录；补 `libxcb-cursor.so.0`。先加载系统 Qt 库会产生符号冲突，改为 PySide6 自带 Qt 优先。系统 Fcitx Qt 插件 ABI 不匹配，移除测试副本，使用 Qt 自带 IBus 插件。
5. 集成真实 LoRA 推理与原技能执行器，执行前重新校验意图和验收凭据，保留原生 CLI 路径。
6. 初版功能通过后发现深色桌面调色板导致文字不清，显式设置浅色文字/背景；缩小窗口发现参数重叠，增加可滚动侧栏；重复任务清空旧日志、参数、阶段与状态栏。
7. 最终九项真实交互全部通过，检查截图、连续像素变化、物理重放、工作进程回收与冻结完整性。

## 最终检查命令

```bash
cd /home/smgbro/mujoconew/GITHUB

# 幂等环境检查/补齐；本机已经装好
bash scripts/132_setup_qt_runtime.sh
bash scripts/128_launch_qt.sh

# 真正的原生 MuJoCo 窗口，验证后自动关闭
bash scripts/116_run_stage6_model.sh '手先到杯子边上去' --scene first --execute --verify-render

# 图形端全链路检查，每次必须用新的 output 目录
env LD_LIBRARY_PATH=/media/smgbro/shared/lora/language/gui-runtime/packages/PySide6/Qt/lib:/media/smgbro/shared/lora/language/gui-runtime \
  PYTHONPATH=/media/smgbro/shared/lora/language/gui-runtime/packages:src \
  QT_QPA_PLATFORM=xcb QT_IM_MODULE=ibus \
  data/runtime/stage6-study-venv/bin/python scripts/129_check_qt_grasp.py \
  --output data/processed/qt_desktop_v1/gui-delivery

# 单元测试与冻结校验
env PYTHONPATH=src:scripts \
  LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/smgbro/miniconda3/envs/dexmv/bin/python -m unittest discover -s tests
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/122_verify_guard_revision.py verify

# 复核原始产物、运行质量检查并发布两张图与小型 JSON
python3 scripts/133_publish_qt_evidence.py \
  --gui-run data/processed/qt_desktop_v1/gui-delivery \
  --native-run /media/smgbro/shared/lora/language/desktop_runs/20261002T133142878689Z
```

`gui-delivery` 已存在时不要覆盖；新验收改用新目录并相应传给发布器。最小 GL probe 脚本是 `124_probe_mujoco_render.py`；应在 `runtime.physics_environment()` 返回的独立环境运行，不要在当前 shell 全局修改库路径。

## 验收与残余限制

- 两次完整搬运、一次仅接近、运行中停止、越界拒绝、仅检查计划、明确停止、规划中停止、关闭窗口共 9 项通过。
- 三次完整计划的动作重放误差均 0；全部实际动作由 `env.step` 执行，初始化一次、执行中位姿写入零。
- 第一/第二搬运最大穿透 0.577/0.901 mm，仍保留 1 mm 原验收上限；没有放宽门槛。
- 1360×880 与 1000×720 窗口检查通过；有滚动条的小窗口不要求同时显示侧栏全部参数，但不允许控件重叠。
- 187 项单元测试通过，GUI 心跳最大间隔 89 ms，中文输入法提交事件通过，所有工作进程结束。
- 原生窗口第一视频接近任务 512 步完成，状态重放误差 0。
- 有限指令和标准场景范围不变；不支持倒水、递到人手、未知目标坐标，不是实机系统。停止是停止仿真步进，不是硬件急停。
- 低层不是已训练 DAPG 策略；将学习策略接入技能系统需要独立验收，不在本轮暗中替换。
- 本轮未人工验证拼音候选窗的全部交互，也没有做 Wayland/无头/其他显卡适配。当前依赖 X11 与本机 NVIDIA 驱动。

## 修改清单

- 新增 `src/fromrealhand/desktop/`：进程环境、计划校验、渲染上下文、Qt 窗口。
- 新增 `scripts/124` 至 `133`：探针、推理/仿真 worker、启动、交互验证、原生兼容、依赖安装和证据发布。
- 仅改动旧 `scripts/116_run_stage6_model.sh` 的渲染分支，非渲染入口仍调用冻结旧脚本。
- 新增两组共 12 项单元测试、Qt 精简依赖清单、操作与答辩资料；README 和网页看板增加入口。

精确原始结果、命令输出、源码 SHA256、运行位置和两张截图见 `docs/presentation/qt_desktop/evidence/`；共享盘交付副本见 `/media/smgbro/shared/lora/language/desktop-delivery-v1/`。
