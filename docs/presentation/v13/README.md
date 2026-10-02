# v13：残差策略分工与闭合初期纠偏

本轮只在原 35 个开发工况比较，不读取旧留出集，也不启动长轮次强化学习。减少截图，以结构化指标和代码说明为主。

## 对照设计

| 方法 | 网络分工 | 不变项 |
|---|---|---|
| 共享 | v12 的一个残差网络 | 35 条示范、84 维特征、精确阶段采样、150 轮 BC |
| 分视频 | 两个独立网络，按已知视频身份选择 | 同一参考动作、残差限幅、归一化和采样预算 |
| 分阶段 | 准备、接近、闭合、抬杯、搬运五个网络 | 同一物理参数，阶段边界前后各 0.1 秒平滑混合 |

每个网络宽度仍为 64×64，分工网络总参数量增加，因此不是等参数量对照。路由由已知视频身份或源视频时钟确定，不是语言模型，也没有学习一个自动选择专家的门控网络。共享网络原先已包含视频身份，分视频方案检验独立参数是否能减少干扰。

可训练均值网络参数量分别为 11,550、23,100、57,750。两个新版本均采样 7,523,400 帧次，使用同一训练种子和全局输入/输出归一化。分阶段边界的混合会略微改变各子网络实际承担的权重，具体值见各模型的 `input.json`，不改变总体阶段采样权重。

开发工况中的 `former_test_*` 是此前已转入 v11 开发集的历史场景，本轮沿用原 35 条准入示范；没有重新读取 v11/v12 独立留出结果选取纠偏工况。

## 纠偏准入

1. 完成开发对照后，按两视频等权任务级、严格级通过率与目标误差选择学生。
2. 从第二视频未严格通过的开发工况中，先选任务失败，再按闭合段杯子误差选最多四个。
3. 学生从正常复位运行到闭合初期，平滑切换到已验证的该工况专家动作，加小幅有界手状态反馈。仅发送手的动作，不写运行中的杯子状态。
4. 整条混合轨迹必须通过任务物理门槛、保存动作重放与半步长检查；严格抓法门槛另外报告。失败轨迹不收录。
5. 只收录完全接管后至抬杯初段的实际执行动作，不把未执行的专家建议标成正确标签。
6. 若有合格数据，进行 20 轮小规模微调，同时运行同样样本预算的无纠偏微调对照；评价仍是开发结果，不宣称独立泛化收益。

专家接管成功不代表学生能自行纠偏。标签来自仿真程序专家，不是真实视频接触力或新的人类示范。

保留的共享基线在候选失败工况中，闭合段杯子位置 RMSE 约 1–2 mm，超过 10 mm 的偏离多发生在后续抬杯/搬运。因此这里的闭合纠偏是对小偏离的预防性接管，不宣称已实现明显滑落后的恢复。v12 中闭合刚开始就失稳的典型案例属于未采用的 130 维参考/接触组合网络，不能混为同一条失败证据。

## 关键代码

`src/fromrealhand/routed_residual.py`：

```python
predictions = torch.stack([head(features) for head in self.heads], dim=1)
residual = (predictions * routes[:, :, None]).sum(dim=1)
action = reference[step] + np.clip(residual, -limits, limits)
```

纠偏标签满足 `label + reference[step] == executed_action`。手状态跟踪反馈必须在完整自由物体物理轨迹上通过验证，不能仅凭关节接近或预测损失准入。

## 运行入口

使用现有 `dexmv` 环境和项目记录的 MuJoCo 库路径：

```bash
python scripts/83_run_v13_routing.py train --mode video
python scripts/83_run_v13_routing.py train --mode phase
python scripts/83_run_v13_routing.py develop --mode shared
python scripts/83_run_v13_routing.py develop --mode video
python scripts/83_run_v13_routing.py develop --mode phase
python scripts/83_run_v13_routing.py compare
python scripts/84_collect_v13_corrections.py collect
python scripts/84_collect_v13_corrections.py finetune
python scripts/83_run_v13_routing.py develop --mode base_finetune
python scripts/83_run_v13_routing.py develop --mode corrected
```

新输出位于 `data/processed/dual_video_v13/`，脚本拒绝覆盖旧训练与纠偏结果。协议为 [v13-study.json](../../../configs/v13-study.json)。

- [开发结果与纠偏准入](RESULTS.md)
- [最终结构化核验](evidence/verification.json)
- [开发集候选选择](evidence/development-selection.json)，尚非独立测试结论

用 `PYTHONPATH=src python scripts/85_build_v13_evidence.py` 重新核验哈希、物理验收结果、阶段路由、实际执行标签和全部单元测试。本轮不新增截图。

## 原生仿真窗口

在 Ubuntu 桌面终端查看本轮开发集首选候选的第二视频名义工况：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/31_view_video_faithful_gpu.sh \
  --rollout data/processed/dual_video_v13/base_finetune/rollouts/second/nominal.pkl \
  --geometry data/processed/dual_video_v10/development/nominal/geometry.npz \
  --simulation-only --episodes 1
```

显示的是已保存策略动作经物理引擎重放，不是在线网络推理，也不逐帧写入杯子位姿。绿色透明杯子仍是无碰撞目标标记。本轮没有重新检查桌面窗口，不以截图数量作为验收。
