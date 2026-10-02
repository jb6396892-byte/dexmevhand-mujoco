# Qt 抓杯实验台

2026-10-02。本机已安装和验证，直接从 Ubuntu 桌面终端启动：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/128_launch_qt.sh
```

不需要另外激活 conda，也不需要启动网页服务。共享盘必须挂载在 `/media/smgbro/shared`；不要用 `sudo` 启动界面。

## 如何查看效果

1. 左侧选择第一或第二视频的标准场景。
2. 输入“把杯子搬到目标位置”，点击“检查计划”，查看技能计划及“模型输出”页签。
3. 点击“执行”。高层重新生成并验证计划，随后画面连续显示接近、闭合、抬杯、搬运。
4. 观察杯底高度、目标距离、当前/峰值穿透、五指法向力、关节越限、动作幅值及阶段进度。曲线分别显示高度和目标距离。
5. 使用正面/侧面/俯视和视距滑块检查手杯接触；右上角保存按钮可截取当前界面，“运行记录”打开本次 JSON 与物理数据目录。
6. 点击“停止”可停止仿真步进，关闭窗口也会清理子进程。下一次执行从所选标准场景的初态开始，不是接着停止位置继续执行。

窗口可以缩小，左侧内容不足时滚动查看，不会把接触参数压在一起。结束后保留最终帧；这是实时物理仿真，不是 MP4 播放器。

### 建议验收

| 操作 | 应看到的结果 |
| --- | --- |
| 第二视频：“把杯子搬到目标位置” | 四阶段完成，杯子离桌并到目标附近 |
| 第一视频：“抓起杯子” | 计划是 reach、grasp、lift，不附加搬运 |
| “手先到杯子边上去” | 只完成 reach，不要求抬杯 |
| “将杯子放到我手上” | 已拒绝，0 步，不创建仿真 |
| “停止” | 已停止，0 步，不创建仿真 |
| 执行中点击停止或关闭窗口 | 停止后不继续步进，无残留工作进程 |

“检查计划”不执行物理动作。合法但暂未覆盖的中文表达可能被拒绝；不能把有限测试中的零误接受理解为任意语言的安全保证。

## 模型与架构

```text
Qt / PySide6 窗口
  -> 独立语言进程：Qwen2.5-0.5B-Instruct + LoRA v4 / guard2
  -> 原始模型 JSON + Schema + 指令语义与低层可行性检查
  -> 独立物理进程再次验证计划、模型验收凭据与文件一致性
  -> 技能执行器：reach -> grasp -> lift -> transport
  -> 已验证专家参考动作 -> env.step(action) -> MuJoCo
  -> 实时 GPU 渲染帧 + 接触/高度/距离等参数 -> Qt
```

高层确实调用训练好的语言模型；没有用规则输出冒充模型结果。低层沿用阶段 5/6 已验收的专家参考动作，**不是第 200 次 DAPG 策略，也不是新训练的独立技能网络**。两标准场景的测试不代表新初态或实物泛化。小指没有持续接触，视频没有真实接触力标签，显示的力均来自 MuJoCo。

Qt 不导入 torch 或 mujoco-py，通过 `QProcess` 和有大小上限的 JSONL 通信。规划和仿真分进程，避免 Python 3.9/3.7、Qt/GL 及不同依赖互相污染。仿真只在初始化时恢复一次完整状态，后续动作直接使用归一化控制，不二次缩放，不逐帧设置手或杯子的位姿。

## Missing GL version 修复

实测旧启动环境会导入 CPU/OSMesa 版 `mujoco-py`，打开 GLFW 窗口时报 `GLEW initalization error: Missing GL version`。只增加 PRIME 两个变量不足以改变导入的扩展。

本次修复限定在启动器和新进程：

- 显式选择已有 `.local/mujoco-py-gpu` GPU 扩展，设置 NVIDIA PRIME offload、GLX vendor 以及 GLEW/GL 加载顺序。
- Qt 使用自己的 Qt 6.8.3 库，语言进程只加载语言依赖，MuJoCo 进程清除 Qt 与强制 CPU/软件渲染变量。
- 隐藏渲染窗口采用双缓冲 GLX 上下文，解决旧版 GLFW 离屏默认单缓冲在本机出现的 `Failed to find a suitable GLXFBConfig`；实际图像仍从 MuJoCo FBO 读取。
- Qt 缺少的 `libxcb-cursor` 只解包放在共享盘隔离目录，没有修改系统驱动。中文输入使用 Qt 自带 IBus 插件，不混入系统不同版本的 Fcitx Qt 插件。

未重新编译或替换已可用的 GPU 二进制，未更改原 `dexmv` 环境、接触参数、碰撞几何、示范或 checkpoint。原生窗口入口也已修复：

```bash
bash scripts/116_run_stage6_model.sh '抓起杯子' --scene second --execute --render
```

参考：[MuJoCo-py 同类 GLEW 问题](https://github.com/openai/mujoco-py/issues/408?timeline_page=1)、[上下文实现](https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjrendercontext.pyx)、[NVIDIA PRIME 文档](https://download.nvidia.com/XFree86/Linux-x86_64/535.261.03/README/primerenderoffload.html)、[Qt QProcess](https://doc.qt.io/qt-6/qprocess.html)、[Qt Linux 依赖](https://doc.qt.io/qt-6/linux-requirements.html)、[Fcitx 应用通信协议](https://fcitx-im.org/wiki/How_does_an_application_talk_to_Fcitx)。这是工程集成与渲染修复，不是新的抓取算法或模型训练。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 真实 Qt + 模型 + 仿真交互 | 9/9 通过 |
| 第一视频完整搬运 | 1417 步，目标误差 0.853 mm，最大穿透 0.577 mm |
| 第二视频完整搬运 | 1450 步，目标误差 3.858 mm，最大穿透 0.901 mm |
| 动作重放一致性 | 最大状态误差 0，执行中状态写入 0 |
| 窗口与连续图像 | 1360×880、1000×720；连续帧非空，布局无重叠 |
| GPU | RTX 4060 Laptop，OpenGL 4.6.0 NVIDIA 595.91.07 |
| UI 响应 | 50 ms 心跳最大间隔 89 ms，无残留工作进程 |
| 中文输入 | Unicode 输入法提交事件通过；尚未人工逐项测试拼音候选操作 |
| 项目回归 | 187 项单元测试通过，冻结语言/技能凭据通过 |

本机约 12 帧/秒显示，仿真按原 10 ms 控制步长推进，不改变动作时序。闭合/抬杯/搬运的物理判据仍使用冻结技能注册表；1 mm 是数值穿透验收上限，不是宣称零穿透。

截图与机器可读报告见 [答辩材料](presentation/qt_desktop/README.md)，完整检查命令见 [运行记录](run_logs/2026-10-02-qt-desktop.md)。

## 文件位置

| 内容 | 位置 |
| --- | --- |
| Qt 源码 | `src/fromrealhand/desktop/` |
| 启动与验证 | `scripts/124_*.py` 至 `scripts/133_*.py` / `.sh` |
| Qt 隔离依赖 | `/media/smgbro/shared/lora/language/gui-runtime/` |
| LoRA 与基础模型 | 原 `language/study_v4_guard2/`、`language/model/`，未移动 |
| 每次交互的原始输出 | `/media/smgbro/shared/lora/language/desktop_runs/<时间戳>/` |
| 小型报告和两张截图 | `docs/presentation/qt_desktop/` |
| 共享盘资料副本 | `/media/smgbro/shared/lora/language/desktop-delivery-v1/` |
| 阶段 3 产物 | 保持项目原位置，没有放进 lora |

环境缺失时运行 `bash scripts/132_setup_qt_runtime.sh`，仅安装 PySide6-Essentials 6.8.3、shiboken6 和隔离 xcb 库，不会安装或升级显卡驱动。依赖当前本机已有的阶段 6 Python 引导环境和 GPU MuJoCo 扩展，不是任意新电脑的一键部署器。

### 常见启动问题

- 提示共享盘或 Qt runtime 不存在：先确认硬盘挂载、模型与 gui-runtime 文件完整，不要在 Linux 盘重复下载模型。
- 提示没有 DISPLAY：从 Ubuntu 图形桌面的终端启动；纯 SSH/无显示服务器不在本轮支持范围。
- 修改过训练文件后“execution locked”：先恢复正确冻结版本或重新做验收，不绕过校验。
- 中文输入异常：先切换本机 Fcitx5 输入法；启动器默认用兼容的 `ibus` Qt 插件。不要把系统 Qt 插件复制到 PySide6 目录。
- 不要在 conda 中直接运行 GUI Python 或把系统 Qt 库放到其前面；统一使用 `128_launch_qt.sh`。
