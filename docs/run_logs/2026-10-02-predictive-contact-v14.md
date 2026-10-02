# v14：接触预测修正与长训练准入

工作目录：`/home/smgbro/mujoconew/GITHUB`。分支：`video-faithful-v1`。

## 本轮目标

延续 v13 分视频/分阶段及纠偏对照，优先解决第二视频的接触峰值问题，完成可以开始受控长训练的工程检查。实际长训练必须先通知用户，本轮不启动。

## 决策与方法

- 保留 v13 最佳共享残差 BC。分工与增加纠偏数据没有胜过这个基线，不强行切换模型。
- 查询 MuJoCo MPC、预测式安全过滤、DexMachina、DexTrack 和原始 MJRL DAPG，来源与采用边界见 [方法说明](../presentation/v14/README.md)。
- 在独立 MuJoCo 数据副本中预测未来参考动作，只对手部执行器动作进行有限候选搜索。未改变物体动力学参数、未施加物体辅助力、未逐帧写入真实杯子状态。
- 保留 v14、v14b 两个 2/4 负结果；v14c 加入有界坐标候选后，小试 4/4，开发 35/35。
- 修复旧 API 可选数组为 `None` 的分支复制兼容性，并审计 warm start、动作单次缩放和副本/真实执行的一步一致性。
- 将单头路由均值网络转换成原生高斯策略接口，50,156 个输入上的均值最大误差为零。采样器两视频名义回放的动作和观测差异均为零。
- 本机旧 DAPG 内部硬编码更新尺度，在项目内加入实测 KL 回溯上限 0.002，未修改外部学习库。

## 已完成的独立评估

冻结提交 `a57b33bb362a619acc67e8ec71e5448334a73d4b` 在测试前已推送并核对远端。冻结策略 SHA256：

```text
7c797b5823c34ccddf0fa861c77bdb0714c783a50e41dbe108d37934d31596b8
```

开发任务级由 29/35 到 35/35，严格级由 27/35 到 33/35。新留出任务级由 11/16 到 15/16，严格级由 6/16 到 7/16，两者稳定抬杯均为 16/16。第二视频任务级由 3/8 到 7/8。

两视频名义工况半步长均通过任务级和严格级。保留第二视频穿透超限、无名指接触丢失和精度失败，不用测试失败再调本轮策略。新留出 32 次配对回放耗时 542.54 秒。

## 主要命令

环境采用 `dexmv`、单线程 BLAS，MuJoCo 库路径及原生窗口命令见 [COMMANDS](../presentation/v14/COMMANDS.md)。

```bash
python scripts/86_v14_contact_readiness.py pilot --config configs/v14-study.json
python scripts/86_v14_contact_readiness.py pilot --config configs/v14b-study.json
python scripts/86_v14_contact_readiness.py pilot --config configs/v14c-study.json
python scripts/86_v14_contact_readiness.py develop --config configs/v14c-study.json
python scripts/86_v14_contact_readiness.py freeze --config configs/v14c-study.json
# 提交冻结凭据后：
python scripts/86_v14_contact_readiness.py heldout --config configs/v14c-study.json
python scripts/87_train_v14_dapg.py --config configs/v14c-study.json
python scripts/88_build_v14_evidence.py --config configs/v14c-study.json
```

## 修改文件与产物

- `src/fromrealhand/predictive_contact.py`：独立副本预测、候选搜索、动作及状态审计。
- `src/fromrealhand/contact_training.py`：真实接触代价与实测 KL 回溯。
- `src/fromrealhand/residual_sampling.py`：将预测动作修正接入实际采样，保留原始残差概率密度契约。
- `scripts/86` 至 `scripts/89`：开发/冻结/留出、短训、自动证据构建、原生物理回放入口。
- `scripts/31_view_video_faithful.py`：支持少量指定关键帧，避免自动生成多余截图。
- `configs/v14*.json`、`tests/test_predictive_contact.py`：协议、训练参数、四项新单元测试。
- `data/processed/dual_video_v14*`：本机轨迹和策略，不上传大数据或许可模型。
- `docs/presentation/v14`：结果、三张关键帧、8 页答辩提纲、代码片段、结构化证据。

## 验收解释

最终自动核验为 `ready_for_long_training=true`，27/27 检查通过；97 项单元测试全部通过，三个关键帧非空且物理动作重放观测误差为零。20 次 DAPG 更新全部有限、非零，采样 57,340 步，最大实测 KL 0.000636；探索轨迹任务级 39/40，训练后两个名义工况均通过任务与严格级，目标误差 0.601/4.627 mm。20 次短训及前后检查耗时 1,446.81 秒。

已停在长训练前。旧策略、演示与 checkpoint 保留，短训末次策略没有冒充独立留出候选。下一步建议先执行 200 次受控长训练并评估，再考虑 2000 次；按当前负载粗估分别约 3.7 小时和 36.6 小时。

格式、策略能运行、任务级成功、严格视频抓法复现是不同检查。最终是否准入以 `docs/presentation/v14/evidence/readiness.json` 为准。

剩余边界包括两个独立视频覆盖不足、部分接触安全余量小、无名指在一个新工况丢失接触、指尖误差、半步长敏感性、预测仿真 CPU 开销，以及没有实物力标签。共享盘未连接，本轮不声称重新验证原视频/MANO 或新视频姿态估计。
