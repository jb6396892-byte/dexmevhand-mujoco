# 第 200 次策略：预注册独立泛化评估

评估前固定 `configs/stage3-post200-eval-v1.json`，两已知视频各 16 个新随机工况。
对照为冻结的训练前策略与固定第 200 次 checkpoint，二者均保留同一预测接触修正器。
不选择中间 checkpoint，不改任务/严格门槛，不据测试结果重新训练。

阶段 3 产物：`data/processed/dual_video_v14c/post200_independent_v1/`。
原训练模型仍在 `data/processed/dual_video_v14c/long_training/`，未移动或覆盖。
共享盘 `/media/smgbro/shared/lora/` 仅用于阶段 6 语言模型、隔离环境和训练产物。
目录按用户澄清在完成 10/64 次回放后调整，冻结协议与两模型哈希保持不变，
保留全部已完成结果并从断点继续；未完成的第 11 次重新执行。
`protocol-v1.json` 是运行前冻结副本，结果将在评估完成后另行存档。

范围仅为已知两视频下新的合成位置、目标与朝向组合，不等于新视频、人类受试者或实物迁移。
