# 200 次 DAPG 训练与监督

**最终状态（2026-10-02 18:23:19 北京时间）：200/200 已完成，退出码 0。**
200 次均有非零更新；训练后两标准场景通过任务与旧严格门槛，目标误差 1.579 / 3.722 mm。
新的独立留出已完成：训练后任务 30/32，对照 31/32；不退化门槛未通过，不替换原策略。
详见 [独立评估及两张截图](presentation/post200/README.md)。最终 [状态](presentation/stage6/evidence/v1/training-status.json)、
[名义评估](presentation/stage6/evidence/v1/training-summary.json)、[5 次训练采样失败](presentation/stage6/evidence/v1/training-failures.json) 已存档。
下面的后台管理说明为运行期间记录，不表示当前仍在训练。

用户授权后于 **2026-10-02 14:42:17（北京时间）** 启动。总预算是 200 次策略更新，不是 200 个视频或 2000 次。每次采样两条物理轨迹，来自两个视频的开发工况。

本轮从已完成独立评估的冻结 BC 残差策略出发，保留 150 ms 预测接触修正；不从未经过新留出验证的 20 次短训末尾策略继续。启动源码提交为 `28453ed`，冻结策略 SHA256 为 `7c797b5823c34ccddf0fa861c77bdb0714c783a50e41dbe108d37934d31596b8`。

## 后台运行

训练由 systemd 用户服务 `fromrealhand-v14c-200.service` 运行，监督器是 `scripts/90_supervise_training.py`。关闭当前终端或本次对话不会主动终止这个服务；不要关机、重启或让电脑休眠。没有配置跨重启自动恢复，也没有完整优化器断点续训支持。

运行后端沿用已经验收的配置：**MuJoCo 采样和 actor/DAPG 更新在 CPU，价值网络在 CUDA GPU**。没有在长训中途迁移算法后端。少量阶段 4 验证会共享 CPU，但不修改训练数据或代码。

查看实时状态：

```bash
cd /home/smgbro/mujoconew/GITHUB
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/90_supervise_training.py status

# 每 10 秒刷新，Ctrl+C 只退出查看，不停止训练。
watch -n 10 /home/smgbro/miniconda3/envs/dexmv/bin/python scripts/90_supervise_training.py status

systemctl --user status fromrealhand-v14c-200.service
journalctl --user -u fromrealhand-v14c-200.service -n 20 --no-pager
```

确需停止时使用 `systemctl --user stop fromrealhand-v14c-200.service`，已有 checkpoint 会保留；不要直接再次启动到同一个输出目录。

## 监督规则

| 情况 | 行为 |
|---|---|
| NaN/Inf、实测 KL 超过 0.002 | 自动停止，记录原因 |
| 连续三次参数更新为零 | 自动停止 |
| 900 秒没有完成新迭代 | 按停滞停止 |
| 冻结代码、配置或策略哈希变化 | 自动停止，避免训练中途变更 |
| 手与场景穿透超过 5 mm | 按严重物理异常停止 |
| 普通任务失败、近期五次任务通过率低于 50% | 告警，保留失败轨迹统计，不立即杀掉正常探索 |

**5 mm 仅为异常停机阈值，不是抓取验收阈值。任务验收仍是 1 mm。** 一次 1.1 mm 穿透仍然判定任务失败，即使训练没有被停止。

## 输出与完成判定

- `data/processed/dual_video_v14c/supervision_200/launch.json`：启动命令、冻结哈希、后端与源码提交。
- 同目录 `status.json`：每十秒原子更新的实时状态；`training.log`：训练原始输出。
- `data/processed/dual_video_v14c/long_training/iterations.json`：逐迭代任务/严格检查、奖励、KL、更新幅度、接触代价。
- 同目录 `iteration_0001.pickle`、`0005`、`0020`，之后每 20 次保存一次，最终 `0200`。
- 同目录 `summary.json`：200 次完成后两条名义场景物理回放与训练总结。

`state=running` 只表示在运行；必须同时检查 `state=completed`、`completed_iterations=200` 以及训练后名义结果。自动监督不等于训练后性能一定改善；最终不会自动覆盖冻结策略。名义评估不是新留出评估；旧训练收据中的 `independent_evaluation_done=false` 保持历史原样，新评估另存 `post200_independent_v1/summary.json`。

网页和 GitHub 的历史 [进度快照](presentation/stage4/evidence/training-progress-snapshot.json) 不是实时监控，以文件中的 `updated_at` 为准。本轮实际约 3 小时 41 分钟完成；阶段 4/5 旧快照保留，不覆盖历史证据。
