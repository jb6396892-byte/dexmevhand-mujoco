# 2026-10-02 阶段 4 v2 交付记录

## 范围

用户确认按仿真接触、三指支撑、抬杯和搬运口径继续。按 4.1 至 4.5 依次执行；不修改正在运行的 v14c 200 次训练，不读取新的留出集，不启动独立技能训练。详细材料见 [结果与答辩提纲](../presentation/stage4/V2_RESULTS.md)。

## 实际命令及结果

所有 Python 命令使用 `dexmv` 环境，并设置 `LD_LIBRARY_PATH`、`LD_PRELOAD`、`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。

| 命令 | 结果 |
|---|---|
| `python scripts/90_supervise_training.py status` | 开发期间多次检查，36 → 42 → 48 → 49/200；服务仍运行，无停止告警 |
| `python scripts/96_stage4_prepare.py pilot` | 两视频名义分段验证通过，8 独立 + 2 连续回放 |
| `python scripts/96_stage4_prepare.py build` | 35 条开发示范筛为 27 条、108 片段；8 条拒绝，有明确原因 |
| `python scripts/97_build_skill_inputs.py` | 122 维输入，训练/内部验证 24/3 条轨迹，归一化只拟合训练，Torch 前向有限；优化步数 0 |
| `python scripts/98_validate_skill_entries.py` | 名义 8/8，扰动 12/32；5 类停止单元检查通过 |
| `python scripts/99_finalize_stage4.py` | 修正渲染初始化顺序后成功，281 文件冻结、2 张截图 |
| `PYTHONPATH=src python -m unittest discover -s tests -v` | 126 项通过，含 MANO 官方模型测试；旧依赖有弃用警告，未升级环境 |
| `python scripts/99_finalize_stage4.py --verify-only` | 子阶段回执、数据来源链和 281 文件哈希全部通过 |
| `ELECTRON_RUN_AS_NODE=1 /usr/share/code/code --check docs/assets/dashboard.js` | 网页脚本语法通过；系统没有独立 node，复用 VS Code 自带运行时，无需安装 |
| `git diff --check` | 通过 |

## 修正和失败留存

- 初次生成训练输入时缺少 MuJoCo 动态库环境；可信 pickle 反序列化会导入 `mujoco_py`。在完整环境下重跑成功，没有改数据或运行时版本。
- 延长闭合/抬杯事件确认至 10 步，第二视频搬运切换 694 → 710，短窗保持率 72% → 88%。动作和物理参数不变。
- 六条短窗不稳轨迹隔离；另两条恢复状态后与原动作状态不一致，仍未定位根因，也未放宽 `1e-7` 门槛。
- 初次截图在第二视频被状态不变断言拦截。查阅官方 `mujoco-py` 源码后确认构造渲染器有 `sim.forward()`，把构造移到恢复初态之前；保留失败目录 `data/processed/stage4_pipeline_v2/delivery-render-rejected-01/`。
- 入口扰动失败包括 15 个入口拒绝、4 个动作耗尽未达事件和 1 个穿透失败。失败样本没有伪装成成功示范或专家纠偏数据。

## 修改文件

- 配置：`configs/stage4-pipeline-v2.json`、`configs/stage5-skill-smoke.template.json`。
- 核心：`src/fromrealhand/skill_pipeline.py`、`skill_inputs.py`、`skill_delivery.py`。
- 脚本：`scripts/96_stage4_prepare.py`、`97_build_skill_inputs.py`、`98_validate_skill_entries.py`、`99_finalize_stage4.py`、`stage4_pipeline_common.py`。
- 测试：`tests/test_skill_pipeline.py`、`test_skill_inputs.py`、`test_skill_delivery.py`。
- 文档：阶段 4 工作表、操作指南、README、网页 C2 工作项、答辩材料与本日志；小型 JSON 和两张 JPEG 位于 `docs/presentation/stage4/evidence/v2/`。

大数据、模型、MANO、动作 pickle、训练权重与缓存仍不上传 GitHub。v1 分段产物、原专家和长训 checkpoint 均保留。

## 未解决项与交付含义

阶段 4 数据/接口工程验收通过，只支持进入阶段 5 的名义小试准备。固定动作不能承受所有 ±0.5 mm 入口变化，两条状态恢复异常仍隔离；内部验证不是新视频泛化。下一步先实现技能条件残差小试与可达接触恢复，而不是直接进行技能长训或任意高层组合。200 次整体任务训练继续由系统服务监督，快照不代表训练完成。
