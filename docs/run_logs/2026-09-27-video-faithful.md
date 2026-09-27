# 视频抓法独立版本验证

版本说明与复现命令见 [VIDEO_FAITHFUL_VERSION.md](../VIDEO_FAITHFUL_VERSION.md)。

## 验收结果

- 数据转换和几何优化：71 个有效源帧全部收敛，加入原模型腱耦合约束及手指控制余量。
- 物理执行：种子 0–3 的窄初态均通过物理抓取检查；不是泛化成功率评估。
- 视频忠实度：未通过。标称末尾指尖均值 19.047 mm，要求 <15 mm；整段均值 22.226 mm，要求 <20 mm；拇指末尾 26.752 mm，要求 <25 mm。
- 训练：未启动。`refined/admission.json` 中 `training_ready=false`。
- 旧版本：`main` 和 `physical-grasp-verified-v1` 保留原基线，检查报告 `baseline_unchanged=true`。

## 本次执行检查

在项目根目录、已配置运行库的 dexmv 环境中：

```bash
python -m unittest discover -s tests -q
bash scripts/31_view_video_faithful_gpu.sh --episodes 1 --speed 4
python scripts/31_view_video_faithful.py --output data/processed/seq_dexycb_001/video_faithful_v1/review_final_cpu
```

- 22 项单元测试通过。遗留 chumpy/NumPy 弃用提示仍存在，没有升级运行环境。
- GPU 桌面窗口创建成功，完成 1184 步并正常退出；保存动作重放最大观测误差 0。
- 软件离屏渲染完成，输出 8 个视频/仿真对照关键帧、拼图及 `comparison.json`；已检查末帧图像。
- GPU 离屏路径报 `Failed to initialize OpenGL`，本次未修复 EGL；它不影响已通过的 GLX 桌面窗口。不要用 GPU 启动脚本的 `--output` 路径生成截图，使用上面的软件渲染命令。

末尾一秒拇指、食指、中指、无名指接触比例均为 100%，小指为 0%。窗口播放的是专家控制器保存的归一化动作，初始化后只经 `env.step(action)` 作用于物理系统，不是新训练神经网络，也不是逐帧写杯子位姿。

## 修改范围

- `src/fromrealhand/video_fidelity.py`：五指对应、物体相对坐标误差、时间映射。
- `scripts/29_build_video_faithful.py`：完整源轨迹约束重定向。
- `scripts/30_validate_video_faithful.py`：闭环动作生成、自由物理重放、独立忠实度验收。
- `scripts/32_refine_video_faithful.py`：有界关节修正试验；没有找到更优通过方案，保留迭代 0。
- `scripts/31_view_video_faithful.py` 及 GPU 启动脚本：独立版本实时窗口与对照图。
- `tests/test_video_fidelity.py`：刚体变换不变性、手指错配检测、源时钟顺序。

数据、MANO 模型及二进制 rollout 不上传 GitHub。原始大数据继续位于共享盘。
