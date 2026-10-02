# 2026-10-02 阶段 6 指令边界修正与收尾

## 范围

用户接受阶段 3 旧稳定版本，本轮仅更新其收尾说明，不修改模型、物理参数或评估。
阶段 6 所有新环境数据、权重、评估及执行日志写入共享盘 `lora/language/`。
原 v2/v3 记录与失败保留，旧 `dexmv` 环境未升级；未下载新的基础模型。

## 执行顺序与结果

1. 检查旧失败，发现 Schema/历史物理门控不检查“原请求是否属于技能能力范围”。
2. 新增 360 条对比训练样本，总训练 1160，开发验证 64；冻结 v4 代码/数据/门槛。
3. 旧模型开发原始正确 60/64；新 LoRA 4 epoch、580 更新、379.94 秒，开发 64/64。
4. v4 新 120 条测试原始正确由 101 提升到 118；错误计划门控后 0。
5. 随后的旧 56 条回归原始正确 54，但合法任务被初版门控误拒绝 10/40，未通过可用率门槛。因此不部署 v4，也不修改其冻结协议。
6. 新建 guard2：只修语法等价识别，保留额外动作/目的地拒绝；LoRA 文件与 v4 哈希完全一致。
7. 旧 64+120+56 行并为 240 行回归，以冻结模型的原始输出重算门控。语义 236/240，错误计划放行 0，合法正确放行 140/140。
8. 冻结全新 100 行测试并实际生成旧/新模型输出。旧语义 80/100、新 94/100；旧 Schema 98/100、新 100/100；原门控越界误接受旧 18、新 6，同一 guard2 后均为 0；合法正确放行均 50/50。
9. 新模型正确计划去重后在 MuJoCo 执行 10 项检查，全部通过。4 项注入错误计划均拒绝，无仿真步。
10. 175 项单元/回归测试通过；旧依赖的 chumpy/NumPy 弃用警告保留，不为消除警告升级环境。

训练按验证损失选第 3 epoch `checkpoint-435`，不是看留出分数选模型。
最终 100 行还有 3 类表达、6 条原始错误：交接到我的手中、杯子和水壶同时抬起、水壶搬运。
均被独立语义门控阻止，不改预测、不用规则计划替换，不继续在该测试上调参。

## 命令

```bash
cd /home/smgbro/mujoconew/GITHUB
export STUDY_ROOT=/media/smgbro/shared/lora
bash scripts/stage6_study_python.sh scripts/118_refine_language.py prepare
bash scripts/stage6_study_python.sh scripts/118_refine_language.py baseline
bash scripts/stage6_study_python.sh scripts/118_refine_language.py train
bash scripts/stage6_study_python.sh scripts/118_refine_language.py heldout
bash scripts/stage6_study_python.sh scripts/118_refine_language.py regression
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/122_verify_guard_revision.py prepare
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/122_verify_guard_revision.py regression
bash scripts/stage6_study_python.sh scripts/122_verify_guard_revision.py heldout
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/122_verify_guard_revision.py accept --capture
PYTHONPATH=src:scripts /home/smgbro/miniconda3/envs/dexmv/bin/python -m unittest discover -s tests
```

已有目录禁止覆盖；这段是本轮操作记录，不是要求原地重复执行。

## 正式入口实测

公共 `scripts/116_run_stage6_model.sh` 已切换至 `123_run_guarded_language.py`。

| 指令/场景 | 结果 | 共享盘日志目录（`study_v4_guard2/instructions/` 下） |
|---|---|---|
| 把杯子搬到目标位置 / second / --execute | 完整搬运，exit 0 | `20261002T123122494136Z` |
| 将杯子放到我手上 / first / --execute | 预检拒绝，exit 2；model_called=false，simulation_created=false，steps=0 | `20261002T123157770881Z` |
| 手先到杯子边上去 / first / --execute | 模型仅生成 reach，物理成功，exit 0 | `20261002T123215197420Z` |

执行使用 headless 模式以自动退出；另以离屏 GPU 渲染保存一张第二视频搬运图，人工查看非空且画面正常。
用户演示可加 `--render` 打开原生窗口并保持至手动关闭；本轮未留下持续运行的窗口或训练进程。

## 修改文件与边界

- 新数据：`configs/stage6-refinement-*.json`、`stage6-guard2-heldout.json`。
- 新模块：`instruction_guard.py`、`refinement.py`、`guard_revision.py`。
- `118` 为对比训练，`119/120` 保留初版冻结实现，`121` 归档，`122` 新门控冻结/评估/验收，`123` 当前推理入口。
- 测试：`test_instruction_guard.py`、`test_guard_revision.py`；旧测试未改。
- README、路线图、网页看板和阶段 6 中文资料更新；旧 v3 材料标记为历史但不改原结果。
- 大文件不提交 GitHub；只提交源代码、配置、小型 JSON 报告及一张截图。

未解决的通用研究问题：任意中文/用户表达保证、新物理场景泛化、学习型技能可行性网络、独立低层技能网络、倒水/放置/交接。它们不属于这次有限指令范围的工程验收。
