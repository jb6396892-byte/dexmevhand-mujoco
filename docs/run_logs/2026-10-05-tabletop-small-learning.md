# 2026-10-05 两视频桌面小规模学习记录

## 授权和范围

用户明确授权推进到小规模泛化训练并直接执行、闭环调试、验收、答辩资料归档和 GitHub 同步。此前 v3 的训练锁不再作为阻碍，但原配置保留不改，新建 `configs/tabletop-dual-v4-training.json`。

训练模型与视觉资料存放 `/media/smgbro/shared/visual_grasp/dual-learn-v4/`。不覆盖阶段 3 历史 checkpoint，不把低层模型放入 `lora`；Qt 中文语言运行日志继续按原约定存放在 `lora/language/desktop_runs`。

## 实际命令

在 `/home/smgbro/mujoconew/GITHUB` 执行：

```bash
bash scripts/137_tabletop_gpu.sh scripts/161_prepare_tabletop_bc.py \
  --config configs/tabletop-dual-v4-training.json --mode routed-residual \
  --start-training --epochs 60 \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/routed-60

bash scripts/137_tabletop_gpu.sh scripts/161_prepare_tabletop_bc.py \
  --config configs/tabletop-dual-v4-training.json --mode shared \
  --start-training --epochs 60 \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/shared-60

bash scripts/137_tabletop_gpu.sh scripts/162_evaluate_tabletop_policy.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/routed-60/candidate.pt \
  --seeds 3 --output /media/smgbro/shared/visual_grasp/dual-learn-v4/dev-routed-60

bash scripts/137_tabletop_gpu.sh scripts/163_freeze_tabletop_candidate.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/routed-60/candidate.pt \
  --development /media/smgbro/shared/visual_grasp/dual-learn-v4/dev-routed-60/evaluation.json \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/freeze.json

bash scripts/137_tabletop_gpu.sh scripts/162_evaluate_tabletop_policy.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/routed-60/candidate.pt \
  --split heldout --freeze /media/smgbro/shared/visual_grasp/dual-learn-v4/freeze.json \
  --seeds 101 102 103 104 105 106 107 108 \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/heldout-routed

python3 scripts/159_check_dual_tabletop_qt.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/qt-structured-final

bash scripts/137_tabletop_gpu.sh -m unittest discover -s tests
git diff --check
```

共享 BC 开发对照替换 checkpoint 和输出目录；专家留出对照去掉 `--checkpoint`。所有训练/评估输出目录禁止覆盖，复跑需新名称。完整执行命令也记录在每组 `evaluation.json` 的各场景 `command` 字段。

## 已定位的问题和处理

- 沙箱中 `nvidia-smi` 无法访问驱动，但宿主机 NVIDIA 595.91.07、RTX 4060 8 GB 正常。GPU 训练使用宿主机设备权限，不重装驱动或 PyTorch。
- 共享绝对动作 BC 两个开发案例在接近阶段触发 1 mm 穿透停机。保留原始失败，不放宽门槛，选用已通过两视频验证的参考条件残差 BC。
- 三套视觉推理与仿真进程并发时 CUDA OOM：首轮 Qt 的两次完整任务、专家种子 102 在零动作时退出。保留原日志，仅资源失败可在相同冻结代码下串行补测，不把失败动作轨迹删掉重抽。
- 冻结脚本初稿有多余右括号，生成冻结文件前即 SyntaxError；修正后才创建冻结记录并启动留出测试，没有模型或测试数据改动。
- 残差模型输出作标准归一化动作截断，每段报告截断步数。物理检查仍逐子步执行；不能用截断掩盖穿透或关节越界。
- 首轮 MLP 残差独立测试 14/16，不能视为全部通过。103 在完整四阶段完成后仍有 20.34 mm 真值误差，105 在搬运中丢失视觉关联。专家基线唯一零动作 OOM 原样补测后合计 16/16；资源失败的原报告保留在汇总的 `infrastructure_retries`。
- 对照训练数据发现残差实际可由关节误差的对角反馈表示，新增结构化 BC，只用原六条训练示范在 GPU 上拟合。非零系数约为 `[1.6, 4.0, 1.0, 0.53333, 0.53333, 0.53333]`，其他为零；没有从模拟器读取这些增益作拟合参数。
- 原测试集不再作为新模型独立测试。第二轮新集合提前登记为 201–204、两视频共八例；旧 103、105 仅做失败回归，来源和统计分开。

## 第二轮命令

```bash
bash scripts/137_tabletop_gpu.sh scripts/166_fit_structured_residual.py \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc

bash scripts/137_tabletop_gpu.sh scripts/162_evaluate_tabletop_policy.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --seeds 3 --output /media/smgbro/shared/visual_grasp/dual-learn-v4/dev-structured

bash scripts/137_tabletop_gpu.sh scripts/162_evaluate_tabletop_policy.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --videos second --split regression --seeds 103 105 \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/regression-structured

bash scripts/137_tabletop_gpu.sh scripts/163_freeze_tabletop_candidate.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --development /media/smgbro/shared/visual_grasp/dual-learn-v4/dev-structured/evaluation.json \
  --fresh-protocol configs/tabletop-dual-v4-fresh-test.json \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/freeze-structured.json

bash scripts/137_tabletop_gpu.sh scripts/162_evaluate_tabletop_policy.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --split heldout --freeze /media/smgbro/shared/visual_grasp/dual-learn-v4/freeze-structured.json \
  --seeds 201 202 203 204 \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/fresh-structured
```

同一新测试集的专家基线去掉 `--checkpoint`，输出 `fresh-expert`。Qt 七项最终测试独立运行，不与两套视觉批量评估同时争用 GPU。

## 感知时序修正

第二轮两种方法均 7/8：第二视频 202 在抬杯时两帧相隔 0.21 s，真实位移 41.45 mm，视觉位移 41.39 mm、当前位置误差 0.067 mm；正常动作越过了原 40 mm 帧间位移检查。没有修改 ICP 阈值或物理门槛，只在 `tabletop-control-candidate.json` 增加 `perception_period_by_phase_s.lift=0.10`，并让 CLI/Qt 调用同一个 `perception_interval`。

修正后的开发输出为 `dev-cadence`（两视频种子 3）；已见失败回归为 `regression-cadence`（第二视频 202、103、105）。最终冻结使用 `configs/tabletop-dual-v4-final-test.json`，输出 `freeze-final.json`。第三组独立种子 301–304 的策略/专家输出分别是 `final-structured`、`final-expert`；仍使用相同的 `structured-bc/candidate.pt`，没有再次拟合权重。

原两轮源码分别在共享盘 `frozen-routed-sources/`、`frozen-structured-sources/`，旧版本所有失败均保留。最终报告不能把三轮测试累加成一次独立泛化实验。

## 证据

最终冻结后策略与专家各 8/8；开发 2/2，三个历史失败回归 3/3。Qt 实际执行两视频抬杯、搬运和三项安全操作共七项全部通过；242 项软件测试通过。汇总脚本返回 `delivery_passed: true`。

```bash
bash scripts/137_tabletop_gpu.sh scripts/163_freeze_tabletop_candidate.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --development /media/smgbro/shared/visual_grasp/dual-learn-v4/dev-cadence/evaluation.json \
  --fresh-protocol configs/tabletop-dual-v4-final-test.json \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/freeze-final.json

bash scripts/137_tabletop_gpu.sh scripts/162_evaluate_tabletop_policy.py \
  --checkpoint /media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt \
  --split heldout --freeze /media/smgbro/shared/visual_grasp/dual-learn-v4/freeze-final.json \
  --seeds 301 302 303 304 \
  --output /media/smgbro/shared/visual_grasp/dual-learn-v4/final-structured

bash scripts/137_tabletop_gpu.sh scripts/164_publish_tabletop_learning.py \
  --expert-infrastructure-retry /media/smgbro/shared/visual_grasp/dual-learn-v4/expert-oom-retry/evaluation.json
```

专家配对命令去掉 checkpoint，使用相同最终冻结文件与种子，输出 `final-expert`。上面的路径记录原运行，禁止覆盖；复跑需要新输出路径。Qt 原始报告 `qt-structured-final/quality.json` 的 `training_started: false` 仅表示界面测试期间没有启动训练，不否定此前已完成的小规模 GPU 训练。

最终结果、阶段偏差和少量截图见 `docs/presentation/tabletop_learning_v4/evidence/`。完整逐步状态与 RGB-D 原始帧保留在共享盘，不提交 GitHub。

代码主要改动：训练入口统计、学习策略加载器、现有物理回放接入、独立评估和冻结脚本、Qt checkpoint 选择、学成策略启动脚本及对应单元测试。现有未提交的桌面视觉控制和 v3 文档属于同一功能链，一并纳入本次交付，不删除或回退。
